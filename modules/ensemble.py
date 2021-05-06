import numpy as np
from torch import Tensor as T
from torch import nn
from typing import List, Optional


class Ensemble(nn.Module):
    def __init__(self, *models: nn.Module, weights: Optional[List[float]] = None):
        super().__init__()
        assert len(models) > 0
        self.models = nn.ModuleList(models)
        self.size = len(self.models)
        if weights is not None:
            assert len(weights) == self.size
            assert np.isclose(sum(weights), 1)
            self.weights = weights
        else:
            self.weights = np.ones(self.size) / self.size

    def forward(self, x: T) -> T:
        out = None
        for model, weight in zip(self.models, self.weights):
            if out is None:
                out = model(x) * weight
            else:
                out += model(x) * weight
        return out
