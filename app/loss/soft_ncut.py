from typing import Optional
from torch import Tensor
import torch.nn as nn
from app.loss.soft_ncut_AsWali import soft_n_cut_loss


class SoftNCutLoss(nn.NLLLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        *_, h, w = x.shape
        return soft_n_cut_loss(target, x, (h, w))
