from brainpy._src.dyn.base import DynamicalSystemNS


class Layer(DynamicalSystemNS):
  """Base class for a layer of artificial neural network."""

  def reset_state(self, *args, **kwargs):
    pass
