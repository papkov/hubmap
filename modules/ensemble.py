import torch
from torch import Tensor as T
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, *models: nn.Module):
        super().__init__()
        assert len(models) > 0
        self.models = models

    def forward(self, x: T) -> T:
        out = None
        for model in self.models:
            if out is None:
                out = model(x)
            else:
                out += model(x)
        out.div_(len(self.models))
        return out
