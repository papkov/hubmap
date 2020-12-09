from typing import Any, Optional

import segmentation_models_pytorch as smp
import torch
from fastai.layers import *
from torch import Tensor, nn
from torch.nn import functional as F


def get_segmentation_model(
    arch: str,
    encoder_name: str,
    encoder_weights: Optional[str] = "imagenet",
    **kwargs: Any
) -> nn.Module:
    """
    Fetch segmentation model by its name
    :param arch:
    :param encoder_name:
    :param encoder_weights:
    :param kwargs:
    :return:
    """
    arch = arch.lower()
    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "unetplusplus" or arch == "unet++":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "linknet":
        return smp.Linknet(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "pspnet":
        return smp.PSPNet(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "pan":
        return smp.PAN(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "deeplabv3":
        return smp.DeepLabV3(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )
    elif arch == "deeplabv3plus" or arch == "deeplabv3+":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs
        )


class UneXt50(nn.Module):
    """
    https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
    """

    def __init__(self, stride=1, **kwargs):
        super().__init__()
        # encoder
        m = torch.hub.load(
            "facebookresearch/semi-supervised-ImageNet1K-models", "resnext50_32x4d_ssl"
        )
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1), m.layer1
        )  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        # aspp with customized dilatations
        self.aspp = ASPP(
            2048,
            256,
            out_c=512,
            dilations=[stride * 1, stride * 2, stride * 3, stride * 4],
        )
        self.drop_aspp = nn.Dropout2d(0.5)
        # decoder
        self.dec4 = UnetBlock(512, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([512, 256, 128, 64], [16] * 4)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = nn.Conv2d(32 + 16 * 4, 1, kernel_size=1)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)
        dec3 = self.dec4(self.drop_aspp(enc5), enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x


class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_ch * 2),
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
                )
                for in_ch, out_ch in zip(input_channels, output_channels)
            ]
        )

    def forward(self, xs: list, last_layer):
        hcs = [
            F.interpolate(
                c(x), scale_factor=2 ** (len(self.convs) - i), mode="bilinear"
            )
            for i, (c, x) in enumerate(zip(self.convs, xs))
        ]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class UnetBlock(nn.Module):
    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        nf: int = None,
        blur: bool = False,
        self_attention: bool = False,
        **kwargs
    ):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = nf if nf is not None else max(up_in_c // 2, 32)
        # self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        # self.conv2 = ConvLayer(
        #     nf,
        #     nf,
        #     norm_type=None,
        #     xtra=SelfAttention(nf) if self_attention else None,
        #     **kwargs
        # )

        self.conv1 = nn.Conv2d(ni, nf, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + [
            _ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4)
            for d in dilations
        ]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
        )
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = torch.nn.functional.interpolate(
            x0, size=xs[0].size()[2:], mode="bilinear", align_corners=True
        )
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
