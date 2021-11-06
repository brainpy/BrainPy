# -*- coding: utf-8 -*-

from brainpy import errors
from brainpy.simulation import utils
from brainpy.simulation.brainobjects.base import DynamicalSystem

__all__ = [
  'NeuGroup',
  'Channel',
  'Soma',
  'Dendrite',
]


class NeuGroup(DynamicalSystem):
  """Base class to model neuronal groups.

  There are several essential attributes:

  - ``size``: the geometry of the neuron group. For example, `(10, )` denotes a line of
    neurons, `(10, 10)` denotes a neuron group aligned in a 2D space, `(10, 15, 4)` denotes
    a 3-dimensional neuron group.
  - ``num``: the flattened number of neurons in the group. For example, `size=(10, )` => \
    `num=10`, `size=(10, 10)` => `num=100`, `size=(10, 15, 4)` => `num=600`.
  - ``shape``: the variable shape with `(num, num_batch)`.

  Parameters
  ----------
  size : int, tuple of int, list of int
    The neuron group geometry.
  num_batch : optional, int
    The batch size.
  steps : tuple of str, tuple of function, dict of (str, function), optional
    The step functions.
  steps : tuple of str, tuple of function, dict of (str, function), optional
    The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
    Variables to monitor.
  name : optional, str
    The name of the dynamic system.
  """

  def __init__(self, size, name=None, steps=('update',), **kwargs):
    # size
    if isinstance(size, (list, tuple)):
      if len(size) <= 0:
        raise errors.BrainPyError('size must be int, or a tuple/list of int.')
      if not isinstance(size[0], int):
        raise errors.BrainPyError('size must be int, or a tuple/list of int.')
      size = tuple(size)
    elif isinstance(size, int):
      size = (size,)
    else:
      raise errors.BrainPyError('size must be int, or a tuple/list of int.')
    self.size = size
    self.num = utils.size2len(size)

    # initialize
    super(NeuGroup, self).__init__(steps=steps, name=name, **kwargs)

  def update(self, _t, _dt, **kwargs):
    """The function to specify the updating rule.

    Parameters
    ----------
    _t : float
      The current time.
    _dt : float
      The time step.
    """
    raise NotImplementedError(f'Subclass of {self.__class__.__name__} must '
                              f'implement "update" function.')


# ----------------------------------------------------
#
#         Conductance-based Neuron Model
#
# ----------------------------------------------------


class Channel(DynamicalSystem):
  """Base class to model ion channels.

  Notes
  -----

  The ``__init__()`` function in :py:class:`Channel` is used to specify
  the parameters of the channel. The ``__call__()`` function
  is used to initialize the variables in this channel.
  """

  def __init__(self, **kwargs):
    super(Channel, self).__init__(**kwargs)


# ---------------------------------------------------------
#
#          Multi-Compartment Neuron Model
#
# ---------------------------------------------------------

class Dendrite(DynamicalSystem):
  """Base class to model dendrites.

  """

  def __init__(self, name, **kwargs):
    super(Dendrite, self).__init__(name=name, **kwargs)


class Soma(DynamicalSystem):
  """Base class to model soma in multi-compartment neuron models.

  """

  def __init__(self, name, **kwargs):
    super(Soma, self).__init__(name=name, **kwargs)
