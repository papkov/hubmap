from typing import Callable, List, Optional

import torch
from pytorch_toolbelt.inference import functional as F
from torch import Tensor as T
from torch import nn


class TTA(nn.Module):
    def __init__(
        self,
        forward_transform: Callable,
        backward_transform: Callable,
    ):
        super().__init__()
        self.forward_transform = forward_transform
        self.backward_transform = backward_transform

    def forward(self, x: T, model: nn.Module) -> T:
        return self.backward_transform(model(self.forward_transform(x)))


class TTACompose(nn.Module):
    def __init__(self, ttas: List[TTA], model: Optional[nn.Module] = None):
        super().__init__()
        self.ttas = ttas
        self.model = model

    def forward(self, x: T, model: Optional[nn.Module] = None) -> T:
        if model is None:
            model = self.model
        out = None
        for tta in self.ttas:
            if out is None:
                out = tta(x, model)
            else:
                out += tta(x, model)
        out.div_(len(self.ttas))
        return out


class TTAIdentity(TTA):
    def __init__(self):
        super().__init__(
            forward_transform=nn.Identity(),
            backward_transform=nn.Identity(),
        )


class TTAHorizontalFlip(TTA):
    def __init__(self):
        super().__init__(
            forward_transform=F.torch_fliplr,
            backward_transform=F.torch_fliplr,
        )


class TTAVerticalFlip(TTA):
    def __init__(self):
        super().__init__(
            forward_transform=F.torch_flipud,
            backward_transform=F.torch_flipud,
        )


class TTARot180(TTA):
    def __init__(self):
        super().__init__(
            forward_transform=F.torch_rot180,
            backward_transform=F.torch_rot180,
        )


class TTARot90CW(TTA):
    def __init__(self):
        super().__init__(
            forward_transform=F.torch_rot90_cw,
            backward_transform=F.torch_rot90_ccw,
        )


class TTARot90CCW(TTA):
    def __init__(self):
        super().__init__(
            forward_transform=F.torch_rot90_ccw,
            backward_transform=F.torch_rot90_cw,
        )


class TTATranspose(TTA):
    def __init__(self):
        super().__init__(
            forward_transform=F.torch_transpose,
            backward_transform=F.torch_transpose,
        )


class TTATransposeRot90CW(TTA):
    def __init__(self):
        super().__init__(
            forward_transform=F.torch_transpose_rot90_cw,
            backward_transform=F.torch_rot90_ccw_transpose,
        )


class TTATransposeRot90CCW(TTA):
    def __init__(self):
        super().__init__(
            forward_transform=F.torch_transpose_rot90_ccw,
            backward_transform=F.torch_rot90_cw_transpose,
        )


class TTATransposeRot180(TTA):
    def __init__(self):
        super().__init__(
            forward_transform=F.torch_transpose_rot180,
            backward_transform=F.torch_rot180_transpose,
        )


class TTAD4(TTACompose):
    """
    D4 symmetry group
    """

    def __init__(self, model: nn.Module):
        super().__init__(
            model=model,
            ttas=[
                TTAIdentity(),  # original
                TTARot90CW(),  # rotated by 90 degrees
                TTARot180(),  # rotated by 180 degrees
                TTARot90CCW(),  # rotated by -90 degrees
                TTATranspose(),  # transposed
                TTATransposeRot90CW(),  # transposed rotated by 90 degrees
                TTATransposeRot180(),  # transposed rotated by 180 degrees
                TTATransposeRot90CCW(),  # transposed rotated by -90 degrees
            ],
        )


class TTAD2(TTACompose):
    """
    D2 symmetry group
    """

    def __init__(self, model: nn.Module):
        super().__init__(
            model=model,
            ttas=[
                TTAIdentity(),  # original
                TTARot180(),  # rotated by 180 degrees
                TTAHorizontalFlip(),  # horizontally-flipped
                TTAVerticalFlip(),  # vertically-flipped tensor
            ],
        )


class TTAD1(TTACompose):
    """
    D1 symmetry group (bilateral, horizontal flip only)
    """

    def __init__(self, model: nn.Module):
        super().__init__(
            model=model,
            ttas=[
                TTAIdentity(),  # original
                TTAHorizontalFlip(),  # horizontally-flipped
            ],
        )


def get_tta(symmetry: str, model: nn.Module) -> nn.Module:
    """
    Get a TTA model with
    :param symmetry: D1 | D2 | D4, otherwise ValueError
    :param model: model to obtain predictions for each TTA
    :return: TTA class initialized with model
    """
    symmetry = symmetry.lower()
    if symmetry == "d1":
        return TTAD1(model)
    elif symmetry == "d2":
        return TTAD2(model)
    elif symmetry == "d4":
        return TTAD4(model)
    else:
        raise ValueError(f"Incorrect symmetry group {symmetry}, should be D1 | D2 | D4")
