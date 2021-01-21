import unittest
from typing import Callable

import torch
from torch import Tensor as T
from torch.nn import Identity, Module

from modules import tta


class MaxIntensity(Module):
    def forward(self, x: T) -> T:
        max_channel, _ = torch.max(x, dim=1, keepdim=True)
        return max_channel


class TestTTA(unittest.TestCase):
    def module_test_identity(self, tta_module: Callable):
        model = MaxIntensity()
        x = torch.rand((4, 3, 128, 128))
        t = tta_module(x, model=model)
        o = model(x)
        self.assertTrue(torch.allclose(o, t))

    def test_horizontal_flip(self):
        self.module_test_identity(tta.TTAHorizontalFlip())

    def test_vertical_flip(self):
        self.module_test_identity(tta.TTAVerticalFlip())

    def test_transpose(self):
        self.module_test_identity(tta.TTATranspose())

    def test_transpose_rot90_cw(self):
        self.module_test_identity(tta.TTATransposeRot90CW())

    def test_transpose_rot90_ccw(self):
        self.module_test_identity(tta.TTATransposeRot90CCW())

    def test_transpose_rot180(self):
        self.module_test_identity(tta.TTATransposeRot180())

    def test_rot180(self):
        self.module_test_identity(tta.TTARot180())

    def test_rot90_cw(self):
        self.module_test_identity(tta.TTARot90CW())

    def test_rot90_ccw(self):
        self.module_test_identity(tta.TTARot90CCW())

    def test_d4(self):
        self.module_test_identity(tta.TTAD4(Identity()))

    def test_d2(self):
        self.module_test_identity(tta.TTAD2(Identity()))

    def test_d1(self):
        self.module_test_identity(tta.TTAD1(Identity()))


if __name__ == "__main__":
    unittest.main()
