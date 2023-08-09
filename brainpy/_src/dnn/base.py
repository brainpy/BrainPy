from brainpy._src.dynsys import DynamicalSystem


__all__ = [
  'Layer'
]


class Layer(DynamicalSystem):
  """Base class for a layer of artificial neural network."""

  def reset_state(self, *args, **kwargs):
    pass

