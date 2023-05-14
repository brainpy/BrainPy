from brainpy._src.dynsys import DynamicalSystemNS


class Layer(DynamicalSystemNS):
  """Base class for a layer of artificial neural network."""

  def reset_state(self, *args, **kwargs):
    pass
