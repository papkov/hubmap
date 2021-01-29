# Implements an idea of symmetric Lovasz loss from here:
# https://www.kaggle.com/iafoss/hubmap-pytorch-fast-ai-starter
# https://www.kaggle.com/iafoss/lovasz
# Codebase copied from Catalyst

from functools import partial
from typing import Any, Callable, List

import torch
import torch.nn.functional as F
from catalyst.contrib.nn.criterion.lovasz import (
    _flatten_binary_scores,
    _lovasz_grad,
    mean,
)
from catalyst.utils.torch import get_activation_fn
from torch import nn
from torch import tensor as T
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


# Focal Tversky adopted from  https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py


def dice(
    outputs: T,
    targets: T,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid",
) -> T:
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    score = (2 * intersection + eps * (union == 0)) / (union + eps)

    return score


def dice_loss(outputs: T, targets: T, **kwargs) -> T:
    return 1 - dice(outputs, targets, **kwargs)


def tversky(
    outputs: T,
    targets: T,
    alpha: float = 0.7,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid",
) -> T:
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    tp = torch.sum(targets * outputs)
    fn = torch.sum(targets * (1 - outputs))
    fp = torch.sum((1 - targets) * outputs)
    dn = tp + alpha * fn + (1 - alpha) * fp
    score = (tp + eps * (dn == 0)) / (dn + eps)

    return score


def tversky_loss(outputs, targets, **kwargs) -> T:
    return 1 - tversky(outputs, targets, **kwargs)


def focal_tversky_loss(
    outputs: T,
    targets: T,
    gamma: float = 0.75,
    **kwargs: Any
):
    return tversky_loss(outputs, targets, **kwargs) ** gamma


class Loss(nn.Module):
    def __init__(self, loss_fn: Callable, element_wise: bool = False, **kwargs: Any):
        super().__init__()
        self.element_wise = element_wise
        self.loss_fn = partial(loss_fn, **kwargs)

    def forward(self, logits: T, targets: T):
        """
        Calculates loss between ``logits`` and ``target`` tensors.
        :param logits: model logits
        :param targets: ground truth labels
        :return: computed loss
        """
        assert logits.shape == targets.shape
        # Iterate over batch dimension if element_wise
        if not self.element_wise:
            logits, targets = [logits], [targets]
        loss = torch.stack([self.loss_fn(o, t) for o, t in zip(logits, targets)])
        return loss.mean()


class FocalTverskyLoss(Loss):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
        alpha: float = 0.7,
        gamma: float = 0.75,
        element_wise: bool = False,
    ):
        """
        "A novel focal Tversky loss function and improved Attention U-Net for lesion segmentation"
        https://arxiv.org/abs/1810.07842
        https://github.com/nabsabraham/focal-tversky-unet

        :param eps: smoothing parameter, default 1e-7
        :param threshold: threshold logits if necessary
        :param activation: activation function
        :param alpha: FN weight, when 0.5 - dice loss
        :param gamma: focal exponent, when 1 - no effect
        :param element_wise: calculate by element in the batch, then average
        """
        super().__init__(
            loss_fn=focal_tversky_loss,
            element_wise=element_wise,
            threshold=threshold,
            activation=activation,
            alpha=alpha,
            gamma=gamma,
            eps=eps,
        )


class DiceLoss(Loss):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
        element_wise: bool = False,
    ):
        """
        Dice loss

        :param eps: smoothing parameter, default 1e-7
        :param threshold: threshold logits if necessary
        :param activation: activation function
        :param element_wise: calculate by element in the batch, then average
        """
        super().__init__(
            loss_fn=dice_loss,
            element_wise=element_wise,
            threshold=threshold,
            activation=activation,
            eps=eps,
        )