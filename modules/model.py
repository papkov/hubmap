from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import segmentation_models_pytorch as smp
import torch

try:
    from adabelief_pytorch import AdaBelief
except ModuleNotFoundError:
    print("adabelief not installed")

from catalyst.contrib.nn import Lookahead, RAdam, Ralamb
from catalyst.metrics.dice import dice
from catalyst.utils.swa import average_weights
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import modules.bottleneck_transformer as botnet


@torch.no_grad()
def find_dice_threshold(model: nn.Module, loader: DataLoader, device: str = "cuda"):
    dice_th_range = np.arange(0.1, 0.7, 0.01)
    masks = []
    preds = []
    dices = []
    for batch in tqdm(loader):
        preds.append(model(batch["image"].to(device)).cpu())
        masks.append(batch["mask"])
    masks = torch.cat(masks, dim=0)
    preds = torch.cat(preds, dim=0)
    for th in tqdm(dice_th_range):
        dices.append(dice(preds, masks, threshold=th).item())
        # dices.append(
        #     np.mean([dice(p, m, threshold=th).item() for p, m in zip(preds, masks)])
        # )
    best_th = dice_th_range[np.argmax(dices)]
    return best_th, (dice_th_range, dices)


def get_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    num_epochs: int,
    plateau: Dict[str, Any],
    eta_min: float = 1e-8,
):
    if name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=eta_min
        )
    elif name == "restarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_epochs // 2,
            T_mult=1,
            eta_min=0,
        )
    elif name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **plateau)
    else:
        raise ValueError

    return scheduler


def get_optimizer(
    name: str,
    model_params: Iterable,
    lr: float = 1e-3,
    wd: float = 0,
    lookahead: bool = False,
):
    if name == "adam":
        base_optimizer = optim.Adam(model_params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        base_optimizer = optim.SGD(
            model_params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True
        )
    elif name == "radam":
        base_optimizer = RAdam(model_params, lr=lr, weight_decay=wd)
    elif name == "ralamb":
        base_optimizer = Ralamb(model_params, lr=lr, weight_decay=wd)
    elif name == "adabelief":
        base_optimizer = AdaBelief(model_params, lr=lr, weight_decay=wd)
    else:
        raise ValueError

    # Use lookahead
    if lookahead:
        optimizer = Lookahead(base_optimizer)
    else:
        optimizer = base_optimizer

    return optimizer


def get_segmentation_model(
    arch: str,
    encoder_name: str,
    encoder_weights: Optional[str] = "imagenet",
    pretrained_checkpoint_path: Optional[str] = None,
    checkpoint_path: Optional[Union[str, List[str]]] = None,
    convert_bn: Optional[str] = None,
    convert_bottleneck: Tuple[int, int, int] = (0, 0, 0),
    **kwargs: Any,
) -> nn.Module:
    """
    Fetch segmentation model by its name
    :param arch:
    :param encoder_name:
    :param encoder_weights:
    :param checkpoint_path:
    :param pretrained_checkpoint_path:
    :param convert_bn:
    :param convert_bottleneck:
    :param kwargs:
    :return:
    """

    arch = arch.lower()
    if (
        encoder_name == "en_resnet34"
        or checkpoint_path is not None
        or pretrained_checkpoint_path is not None
    ):
        encoder_weights = None

    if arch == "unet":
        model = smp.Unet(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "unetplusplus" or arch == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "linknet":
        model = smp.Linknet(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "pspnet":
        model = smp.PSPNet(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "pan":
        model = smp.PAN(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "deeplabv3plus" or arch == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "manet":
        model = smp.MAnet(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    else:
        raise ValueError

    if pretrained_checkpoint_path is not None:
        print(f"Loading pretrained checkpoint {pretrained_checkpoint_path}")
        state_dict = torch.load(
            pretrained_checkpoint_path, map_location=torch.device("cpu")
        )
        model.encoder.load_state_dict(state_dict)
        del state_dict

    # TODO fmap_size=16 hardcoded for input 256
    botnet.convert_resnet(
        model.encoder,
        replacement=convert_bottleneck,
        fmap_size=16,
        position_encoding=True,
    )

    # TODO parametrize conversion
    print(f"Convert BN to {convert_bn}")
    if convert_bn == "instance":
        print("Converting BatchNorm2d to InstanceNorm2d")
        model = batch_norm2instance(model)
    elif convert_bn == "group":
        print("Converting BatchNorm2d to GroupNorm")
        model = batch_norm2group(model, channels_per_group=1)
    elif convert_bn == "bnet":
        print("Converting BatchNorm2d to BNet2d")
        model = batch_norm2bnet(model)
    elif convert_bn == "gnet":
        print("Converting BatchNorm2d to GNet2d")
        model = batch_norm2gnet(model, channels_per_group=1)
    elif not convert_bn:
        print("Do not convert BatchNorm2d")
    else:
        raise ValueError

    if checkpoint_path is not None:
        if not isinstance(checkpoint_path, list):
            checkpoint_path = [checkpoint_path]
        states = []
        for cp in checkpoint_path:
            # Load checkpoint
            print(f"\nLoading checkpoint {str(cp)}")
            state_dict = torch.load(cp, map_location=torch.device("cpu"))[
                "model_state_dict"
            ]
            states.append(state_dict)
        state_dict = average_weights(states)
        model.load_state_dict(state_dict)
        del state_dict

    return model


@torch.no_grad()
def batch_norm2other(
    module: nn.Module, OtherNorm: type, *args: Any, **kwargs: Any
) -> nn.Module:
    """
    Converts BatchNorm2d to GroupNorm, InstanceNorm2d or other class inherited from NormBase
    TODO overwrite module attributes in kwargs and do not copy their values later

    Kudos to Ternaus
    :param module: nn.Module to convert all BatchNorm2d in it
    :param OtherNorm: type of normalization (e.g. InstanceNorm2d without brackets)
    :return: module with converted BatchNorm2d
    """
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        if issubclass(OtherNorm, torch.nn.GroupNorm):
            # Set num channels per group to 16 by default as suggested in the paper
            kwargs_copy = kwargs.copy()  # Copy because of recursion
            num_groups = kwargs_copy.pop(
                "num_groups", max(1, module.num_features // 16)
            )
            channels_per_group = kwargs_copy.pop("channels_per_group")
            if channels_per_group is not None:
                # e.g. to assign one channel to each group
                num_groups = module.num_features // channels_per_group

            module_output = OtherNorm(
                num_channels=module.num_features,
                num_groups=min(num_groups, module.num_features),
                affine=module.affine,
                eps=module.eps,
                **kwargs_copy,
            )
        elif issubclass(OtherNorm, torch.nn.modules.batchnorm._NormBase):
            # Consider others inherited from NormBase
            module_output = OtherNorm(
                num_features=module.num_features,
                affine=module.affine,
                eps=module.eps,
                track_running_stats=module.track_running_stats,
                momentum=module.momentum,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(f"Class {OtherNorm} is not supported")
        for constant in module.__constants__:
            if constant == "affine" or (
                not module.affine and constant in ("weight", "bias")
            ):
                # Do not reassign affine or W and b if module is not affine
                continue
            if hasattr(module_output, constant):
                setattr(module_output, constant, getattr(module, constant))

    for name, child in module.named_children():
        module_output.add_module(
            name, batch_norm2other(child, OtherNorm, *args, **kwargs)
        )

    del module
    return module_output


def batch_norm2instance(module: nn.Module) -> nn.Module:
    """
    Converts BatchNorm2d to InstanceNorm2d

    Kudos to Ternaus
    :param module: nn.Module to convert all BatchNorm2d
    :return:
    """
    return batch_norm2other(module, torch.nn.InstanceNorm2d)


def batch_norm2group(
    module: nn.Module,
    num_groups: int = 32,
    channels_per_group: Optional[int] = None,
) -> nn.Module:
    """
    Converts BatchNorm2d to GroupNorm

    :param module: nn.Module to convert all BatchNorm2d
    :param num_groups: for GroupNorm
    :param channels_per_group: overwrites `num_groups` to set it proportional to `num_channels`
    :return:
    """
    return batch_norm2other(
        module,
        torch.nn.GroupNorm,
        num_groups=num_groups,
        channels_per_group=channels_per_group,
    )


def batch_norm2gnet(
    module: nn.Module,
    kernel_size: int = 3,
    num_groups: int = 32,
    channels_per_group: Optional[int] = None,
) -> nn.Module:
    """
    Converts BatchNorm2d to BNet2d
    "Batch Normalization with Enhanced Linear Transformation"

    :param module: nn.Module to convert all BatchNorm2d
    :param kernel_size: kernel size for grouped convolution in GNet2d
    :param num_groups: for GroupNorm
    :param channels_per_group: overwrites num_groups to set it proportional to num_channels (equal by default)
    :return:
    """
    return batch_norm2other(
        module,
        GNet2d,
        kernel_size=kernel_size,
        num_groups=num_groups,
        channels_per_group=channels_per_group,
    )


def batch_norm2bnet(module: nn.Module, kernel_size: int = 3) -> nn.Module:
    """
    Converts BatchNorm2d to BNet2d
    "Batch Normalization with Enhanced Linear Transformation"

    :param module: nn.Module to convert all BatchNorm2d
    :param kernel_size: kernel size for grouped convolution in EnBatchNorm2d
    :return:
    """
    return batch_norm2other(module, BNet2d, kernel_size=kernel_size)


def batch_norm2bnet_resnet(model: nn.Module, kernel_size: int = 3) -> None:
    """
    Converts inplace BatchNorm2d in ResNet-like encoder of a model
    :param model:
    :param kernel_size:
    :return:
    """
    model.encoder.layer1 = batch_norm2bnet(
        model.encoder.layer1, kernel_size=kernel_size
    )
    model.encoder.layer2 = batch_norm2bnet(
        model.encoder.layer2, kernel_size=kernel_size
    )
    model.encoder.layer3 = batch_norm2bnet(
        model.encoder.layer3, kernel_size=kernel_size
    )
    model.encoder.layer4 = batch_norm2bnet(
        model.encoder.layer4, kernel_size=kernel_size
    )


# Batch Normalization with Enhanced Linear Transformation
class EnBatchNorm2d(nn.Module):
    def __init__(self, in_channel, kernel_size: int = 3, eps: float = 1e-5):
        super(EnBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel, eps=eps, affine=False)
        self.conv = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=in_channel,
            bias=True,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class BNet2d(nn.BatchNorm2d):
    """
    https://arxiv.org/pdf/2011.14150.pdf
    https://github.com/yuhuixu1993/BNET/blob/d812c566a9c204d0503a8c4fa3ca76915483b07e/detection/mmdet/models/backbones/resnet.py
    """

    def __init__(
        self, num_features: int, *args: Any, kernel_size: int = 3, **kwargs: Any
    ):
        kwargs.pop("affine")  # Affine always False
        super(BNet2d, self).__init__(num_features, *args, affine=False, **kwargs)
        self.conv = nn.Conv2d(
            num_features,
            num_features,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=num_features,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(super(BNet2d, self).forward(x))


class GNet2d(nn.GroupNorm):
    """
    https://arxiv.org/pdf/2011.14150.pdf
    https://github.com/yuhuixu1993/BNET/blob/d812c566a9c204d0503a8c4fa3ca76915483b07e/detection/mmdet/models/backbones/resnet.py
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        kernel_size: int = 3,
        **kwargs: Any,
    ):
        super(GNet2d, self).__init__(
            num_groups=num_groups, num_channels=num_channels, eps=eps, affine=False
        )
        self.bnconv = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=num_channels,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.bnconv(super(GNet2d, self).forward(x))


class ConvEnBn2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvEnBn2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        # self.bn = EnBatchNorm2d(out_channel, eps=1e-5)
        self.bn = BNet2d(out_channel, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBn2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# bottleneck type C
class EnBasic(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        channel,
        kernel_size=3,
        stride=1,
        is_shortcut=False,
    ):
        super(EnBasic, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvEnBn2d(
            in_channel,
            channel,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
        )
        self.conv_bn2 = ConvEnBn2d(
            channel,
            out_channel,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )

        if is_shortcut:
            self.shortcut = ConvBn2d(
                in_channel, out_channel, kernel_size=1, padding=0, stride=stride
            )

    def forward(self, x):
        z = F.relu(self.conv_bn1(x), inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z, inplace=True)
        return z


class EnResNet34(nn.Module, smp.encoders._base.EncoderMixin):
    def __init__(self, kernel_size_first=7, **kwargs):
        super().__init__()

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels: List[int] = [3, 64, 64, 128, 256, 512]

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = 5

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3

        self.block0 = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=kernel_size_first,
                padding=kernel_size_first // 2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            EnBasic(
                64,
                64,
                64,
                kernel_size=3,
                stride=1,
                is_shortcut=False,
            ),
            *[
                EnBasic(
                    64,
                    64,
                    64,
                    kernel_size=3,
                    stride=1,
                    is_shortcut=False,
                )
                for i in range(1, 3)
            ],
        )
        self.block2 = nn.Sequential(
            EnBasic(
                64,
                128,
                128,
                kernel_size=3,
                stride=2,
                is_shortcut=True,
            ),
            *[
                EnBasic(
                    128,
                    128,
                    128,
                    kernel_size=3,
                    stride=1,
                    is_shortcut=False,
                )
                for i in range(1, 4)
            ],
        )
        self.block3 = nn.Sequential(
            EnBasic(
                128,
                256,
                256,
                kernel_size=3,
                stride=2,
                is_shortcut=True,
            ),
            *[
                EnBasic(
                    256,
                    256,
                    256,
                    kernel_size=3,
                    stride=1,
                    is_shortcut=False,
                )
                for i in range(1, 6)
            ],
        )
        self.block4 = nn.Sequential(
            EnBasic(
                256,
                512,
                512,
                kernel_size=3,
                stride=2,
                is_shortcut=True,
            ),
            *[
                EnBasic(
                    512,
                    512,
                    512,
                    kernel_size=3,
                    stride=1,
                    is_shortcut=False,
                )
                for i in range(1, 3)
            ],
        )

    def forward(self, x) -> List[Tensor]:
        """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
        """
        stages = [
            nn.Identity(),
            self.block0,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        ]

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features


smp.encoders.encoders["en_resnet34"] = {
    "encoder": EnResNet34,
    "params": {
        "kernel_size_first": 7,
    },
    "pretrained_settings": {},
}


# class UneXt50(nn.Module):
#     """
#     https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
#     """
#
#     def __init__(self, stride=1, **kwargs):
#         super().__init__()
#         # encoder
#         m = torch.hub.load(
#             "facebookresearch/semi-supervised-ImageNet1K-models", "resnext50_32x4d_ssl"
#         )
#         self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
#         self.enc1 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1), m.layer1
#         )  # 256
#         self.enc2 = m.layer2  # 512
#         self.enc3 = m.layer3  # 1024
#         self.enc4 = m.layer4  # 2048
#         # aspp with customized dilatations
#         self.aspp = ASPP(
#             2048,
#             256,
#             out_c=512,
#             dilations=[stride * 1, stride * 2, stride * 3, stride * 4],
#         )
#         self.drop_aspp = nn.Dropout2d(0.5)
#         # decoder
#         self.dec4 = UnetBlock(512, 1024, 256)
#         self.dec3 = UnetBlock(256, 512, 128)
#         self.dec2 = UnetBlock(128, 256, 64)
#         self.dec1 = UnetBlock(64, 64, 32)
#         self.fpn = FPN([512, 256, 128, 64], [16] * 4)
#         self.drop = nn.Dropout2d(0.1)
#         self.final_conv = nn.Conv2d(32 + 16 * 4, 1, kernel_size=1)
#
#     def forward(self, x):
#         enc0 = self.enc0(x)
#         enc1 = self.enc1(enc0)
#         enc2 = self.enc2(enc1)
#         enc3 = self.enc3(enc2)
#         enc4 = self.enc4(enc3)
#         enc5 = self.aspp(enc4)
#         dec3 = self.dec4(self.drop_aspp(enc5), enc3)
#         dec2 = self.dec3(dec3, enc2)
#         dec1 = self.dec2(dec2, enc1)
#         dec0 = self.dec1(dec1, enc0)
#         x = self.fpn([enc5, dec3, dec2, dec1], dec0)
#         x = self.final_conv(self.drop(x))
#         x = F.interpolate(x, scale_factor=2, mode="bilinear")
#         return x
#
#
# class FPN(nn.Module):
#     def __init__(self, input_channels: list, output_channels: list):
#         super().__init__()
#         self.convs = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
#                     nn.ReLU(inplace=True),
#                     nn.BatchNorm2d(out_ch * 2),
#                     nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
#                 )
#                 for in_ch, out_ch in zip(input_channels, output_channels)
#             ]
#         )
#
#     def forward(self, xs: list, last_layer):
#         hcs = [
#             F.interpolate(
#                 c(x), scale_factor=2 ** (len(self.convs) - i), mode="bilinear"
#             )
#             for i, (c, x) in enumerate(zip(self.convs, xs))
#         ]
#         hcs.append(last_layer)
#         return torch.cat(hcs, dim=1)
#
#
# class UnetBlock(nn.Module):
#     def __init__(
#         self,
#         up_in_c: int,
#         x_in_c: int,
#         nf: int = None,
#         blur: bool = False,
#         self_attention: bool = False,
#         **kwargs,
#     ):
#         super().__init__()
#         self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
#         self.bn = nn.BatchNorm2d(x_in_c)
#         ni = up_in_c // 2 + x_in_c
#         nf = nf if nf is not None else max(up_in_c // 2, 32)
#         # self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
#         # self.conv2 = ConvLayer(
#         #     nf,
#         #     nf,
#         #     norm_type=None,
#         #     xtra=SelfAttention(nf) if self_attention else None,
#         #     **kwargs
#         # )
#
#         self.conv1 = nn.Conv2d(ni, nf, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
#         s = left_in
#         up_out = self.shuf(up_in)
#         cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
#         return self.conv2(self.conv1(cat_x))
#
#
# class _ASPPModule(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
#         super().__init__()
#         self.atrous_conv = nn.Conv2d(
#             inplanes,
#             planes,
#             kernel_size=kernel_size,
#             stride=1,
#             padding=padding,
#             dilation=dilation,
#             bias=False,
#             groups=groups,
#         )
#         self.bn = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU()
#
#         self._init_weight()
#
#     def forward(self, x):
#         x = self.atrous_conv(x)
#         x = self.bn(x)
#
#         return self.relu(x)
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#
# class ASPP(nn.Module):
#     def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
#         super().__init__()
#         self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + [
#             _ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4)
#             for d in dilations
#         ]
#         self.aspps = nn.ModuleList(self.aspps)
#         self.global_pool = nn.Sequential(
#             nn.AdaptiveMaxPool2d((1, 1)),
#             nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
#             nn.BatchNorm2d(mid_c),
#             nn.ReLU(),
#         )
#         out_c = out_c if out_c is not None else mid_c
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#         )
#         self.conv1 = nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False)
#         self._init_weight()
#
#     def forward(self, x):
#         x0 = self.global_pool(x)
#         xs = [aspp(x) for aspp in self.aspps]
#         x0 = torch.nn.functional.interpolate(
#             x0, size=xs[0].size()[2:], mode="bilinear", align_corners=True
#         )
#         x = torch.cat([x0] + xs, dim=1)
#         return self.out_conv(x)
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
