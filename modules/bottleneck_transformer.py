from collections import OrderedDict
from typing import Tuple, Optional

import torch
from einops import rearrange, repeat
from torch import einsum, nn
from torch import tensor as T

# translated from tensorflow code
# https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
# https://github.com/lucidrains/bottleneck-transformer-pytorch/blob/main/bottleneck_transformer_pytorch/bottleneck_transformer_pytorch.py

# positional embedding helpers


def relative_to_absolute(x: T) -> T:
    b, h, l, _ = x.shape
    col_pad = torch.zeros((b, h, l, 1), device=x.device, dtype=x.dtype)
    x = torch.cat((x, col_pad), dim=3)
    flat_x = rearrange(x, "b h l c -> b h (l c)")
    flat_pad = torch.zeros((b, h, l - 1), device=x.device, dtype=x.dtype)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1) :]
    return final_x


def relative_logits_1d(q: T, rel_k: T) -> T:
    b, heads, h, w, dim = q.shape
    logits = einsum("b h x y d, r d -> b h x y r", q, rel_k) * dim ** -0.5
    logits = rearrange(logits, "b h x y r -> b (h x) y r")
    logits = relative_to_absolute(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = repeat(logits, "b h x y j -> b h x i y j", i=h)
    return logits


class AbsolutePositionEmbedding(nn.Module):
    """
    Absolute positional encoding
    Recommended for classification (easier), but give little gain for segmentation/detection
    """

    def __init__(self, fmap_size: int, dim_head: int):
        super().__init__()
        self.fmap_size = fmap_size
        self.scale = torch.tensor(dim_head ** -0.5)
        self.height = nn.Parameter(torch.randn(fmap_size, dim_head) * self.scale)
        self.width = nn.Parameter(torch.randn(fmap_size, dim_head) * self.scale)

    def forward(self, q):
        emb_h = rearrange(self.height, "h d -> h () d")
        emb_w = rearrange(self.width, "w d -> () w d")
        emb = rearrange(emb_h + emb_w, "h w d -> (h w) d")
        logits = einsum("b h i d, j d -> b h i j", q, emb) * self.scale
        return logits


class RelativePositionEmbedding(nn.Module):
    """
    Relative positional encoding
    Recommended for segmentation/detection, less relevant for classification
    """

    def __init__(self, fmap_size: int, dim_head: int):
        super().__init__()
        self.fmap_size = fmap_size
        self.scale = torch.tensor(dim_head ** -0.5)
        self.relative_height = nn.Parameter(
            torch.randn(fmap_size * 2 - 1, dim_head) * self.scale
        )
        self.relative_width = nn.Parameter(
            torch.randn(fmap_size * 2 - 1, dim_head) * self.scale
        )

    def forward(self, q: T) -> T:
        q = rearrange(q, "b h (x y) d -> b h x y d", x=self.fmap_size)
        logits_w = relative_logits_1d(q, self.relative_width)
        logits_w = rearrange(logits_w, "b h x i y j-> b h (x y) (i j)")

        q = rearrange(q, "b h x y d -> b h y x d")
        logits_h = relative_logits_1d(q, self.relative_height)
        logits_h = rearrange(logits_h, "b h x i y j -> b h (y x) (j i)")
        return logits_w + logits_h


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        fmap_size: int,
        heads: int = 4,
        dim_head: int = 128,
        position_encoding: Optional[str] = "relative",
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        if position_encoding == "relative":
            self.pos_emb = RelativePositionEmbedding(fmap_size, dim_head)
        elif position_encoding == "absolute":
            self.pos_emb = AbsolutePositionEmbedding(fmap_size, dim_head)

    def forward(self, fmap: T):
        b, c, h, w = fmap.shape
        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=self.heads),
            (q, k, v),
        )

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if hasattr(self, "pos_emb"):
            sim += self.pos_emb(q)

        attention = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return out


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module as in
    https://github.com/Cadene/pretrained-models.pytorch/blob/8aae3d8f1135b6b13fed79c1d431e3449fdbf6e0/pretrainedmodels/models/senet.py#L85
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            OrderedDict(
                {
                    "avg_pool": nn.AdaptiveAvgPool2d(1),
                    "fc1": nn.Conv2d(
                        in_channels, in_channels // reduction, kernel_size=1, padding=0
                    ),
                    "relu": nn.ReLU(inplace=True),
                    "fc2": nn.Conv2d(
                        in_channels // reduction, in_channels, kernel_size=1, padding=0
                    ),
                    "sigmoid": nn.Sigmoid(),
                }
            )
        )

    def forward(self, x: T) -> T:
        return self.se(x) * x


class BottleBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        fmap_size: int,
        out_channels: int,
        proj_factor: int = 4,
        downsample: bool = False,
        heads: int = 4,
        dim_head: int = 128,
        position_encoding: Optional[str] = "relative",
        se_reduction: int = 16,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        # shortcut

        if in_channels != out_channels or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                # TODO Do we need an activation here?
                # activation,
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion
        attention_dim = out_channels // proj_factor

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, attention_dim, 1, bias=False),
            nn.BatchNorm2d(attention_dim),
            activation,
            Attention(
                dim=attention_dim,
                fmap_size=fmap_size,
                heads=heads,
                dim_head=dim_head,
                position_encoding=position_encoding,
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(attention_dim),
            activation,
            nn.Conv2d(attention_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # init last batch norm gamma to zero
        nn.init.zeros_(self.net[-1].weight)

        # final activation
        self.activation = activation

        # SE module
        if se_reduction != 0:
            self.se_module = SEModule(out_channels, reduction=se_reduction)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.net(x)
        if hasattr(self, "se_module"):
            x = self.se_module(x)
        x += residual
        return self.activation(x)


class BottleStack(nn.Module):
    """
    Main bottle stack
    """

    def __init__(
        self,
        *,
        in_channels: int,
        fmap_size: int,
        out_channels: int,
        proj_factor: int = 4,
        num_layers: int = 3,
        heads: int = 4,
        dim_head: int = 128,
        downsample: bool = True,
        position_encoding: Optional[str] = "relative",
        se_reduction: int = 16,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.dim = in_channels
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            in_channels = in_channels if is_first else out_channels
            layer_downsample = is_first and downsample
            layer_fmap_size = fmap_size // (2 if downsample and not is_first else 1)

            layers.append(
                BottleBlock(
                    in_channels=in_channels,
                    fmap_size=layer_fmap_size,
                    out_channels=out_channels,
                    proj_factor=proj_factor,
                    heads=heads,
                    dim_head=dim_head,
                    downsample=layer_downsample,
                    position_encoding=position_encoding,
                    se_reduction=se_reduction,
                    activation=activation,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert (
            c == self.dim
        ), f"channels of feature map {c} must match channels given at init {self.dim}"
        assert (
            h == self.fmap_size and w == self.fmap_size
        ), f"height and width of feature map must match the fmap_size given at init {self.fmap_size}"
        return self.net(x)


def convert_resnet(
    resnet: nn.Module,
    replacement: Tuple[int, int, int] = (1, 0, 0),
    downsample: bool = True,
    in_channels: int = 1024,
    out_channels: int = 2048,
    fmap_size: int = 64,
    proj_factor: int = 4,
    heads: int = 4,
    dim_head: int = 128,
    position_encoding: Optional[str] = "relative",
    se_reduction: int = 16,
    activation: nn.Module = nn.ReLU(),
):
    """
    Replaces bottlenecks in c5 block with attention of any ResNet from torchvision
    :param resnet: network to replace blocks
    :param replacement: scheme of replacement, designed as in the paper:
                        (1, 0, 0) - only the first block, default
                        (0, 0, 1) - only the last block
                        (0, 1, 1) - two last blocks
                        (1, 1, 1) - all the blocks
    :param downsample: if do downsample for the first block
    :param in_channels: input channels for the first block
    :param out_channels: output channels
    :param fmap_size: size of feature map for the first block, input size dependent: fmap_size = input_size / 16
    :param proj_factor: projection factor to reduce dimensionality in attention
    :param heads: number of attention heads
    :param dim_head: dimensionality of an attention head
    :param position_encoding: {"relative", "absolute", None} if use relative or absolute positional encoding
    :param se_reduction: squeeze-and-excitation reduction factor, default 16, not applied if 0
    :param activation: activation function
    :return:
    """
    # Shortcut if replacement is (0, 0, 0)
    if not sum(replacement):
        return resnet

    assert hasattr(resnet, "layer4"), "Can't convert this model, no layer4 (c5)"
    assert len(replacement) == len(
        resnet.layer4
    ), "Replacement length does not match length of block"

    for i, replace in enumerate(replacement):
        if replace:
            is_first = i == 0
            in_channels = in_channels if is_first else out_channels
            layer_downsample = is_first and downsample
            layer_fmap_size = fmap_size // (2 if downsample and not is_first else 1)
            resnet.layer4[i] = BottleBlock(
                in_channels=in_channels,
                fmap_size=layer_fmap_size,
                out_channels=out_channels,
                proj_factor=proj_factor,
                heads=heads,
                dim_head=dim_head,
                downsample=layer_downsample,
                position_encoding=position_encoding,
                se_reduction=se_reduction,
                activation=activation,
            )

    return resnet
