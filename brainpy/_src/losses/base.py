from typing import Optional

from brainpy._src.dnn.base import Layer

__all__ = [
  'Loss',
  'WeightedLoss',
]


class Loss(Layer):
  reduction: str

  def __init__(self, reduction: str = 'mean') -> None:
    super().__init__()
    self.reduction = reduction


class WeightedLoss(Loss):
  weight: Optional

  def __init__(self, weight: Optional = None, reduction: str = 'mean') -> None:
    super().__init__(reduction)
    self.weight = weight
