# Implements an idea of symmetric Lovasz loss from here:
# https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
# https://www.kaggle.com/iafoss/lovasz
# Codebase copied from Catalyst

import torch
import torch.nn.functional as F
from catalyst.contrib.nn.criterion.lovasz import (
    _flatten_binary_scores,
    _lovasz_grad,
    mean,
)
from torch.nn.modules.loss import _Loss


def _lovasz_hinge_flat(logits, targets):
    """The binary Lovasz hinge loss.

    Args:
        logits: [P] Variable, logits at each prediction
            (between -infinity and +infinity)
        targets: [P] Tensor, binary ground truth targets (0 or 1)
    """
    if len(targets) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * targets.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = targets[perm]
    grad = _lovasz_grad(gt_sorted)
    # loss = torch.dot(F.relu(errors_sorted), grad)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


def _lovasz_hinge(logits, targets, per_image=True, ignore=None):
    """The binary Lovasz hinge loss.

    Args:
        logits: [B, H, W] Variable, logits at each pixel
            (between -infinity and +infinity)
        targets: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(
                *_flatten_binary_scores(logit.unsqueeze(0), target.unsqueeze(0), ignore)
            )
            for logit, target in zip(logits, targets)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, targets, ignore))
    return loss


class LovaszLossBinary(_Loss):
    """Creates a criterion that optimizes a binary Lovasz loss.

    It has been proposed in `The Lovasz-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure
    in neural networks`_.

    .. _The Lovasz-Softmax loss\: A tractable surrogate for the optimization
        of the intersection-over-union measure in neural networks:
        https://arxiv.org/abs/1705.08790
    """

    def __init__(self, per_image=False, ignore=None, symmetric: bool = True):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image
        self.symmetric = symmetric

    def forward(self, logits, targets):
        """Forward propagation method for the Lovasz loss.

        Args:
            logits: [bs; ...]
            targets: [bs; ...]

        @TODO: Docs. Contribution is welcome.
        """
        loss = _lovasz_hinge(
            logits, targets, per_image=self.per_image, ignore=self.ignore
        )
        if self.symmetric:
            loss += _lovasz_hinge(
                -logits, 1.0 - targets, per_image=self.per_image, ignore=self.ignore
            )
            loss /= 2

        return loss
