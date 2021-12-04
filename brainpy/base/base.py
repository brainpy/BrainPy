# -*- coding: utf-8 -*-

import logging
import os.path

from brainpy import errors
from brainpy.tools import namechecking
from brainpy.base.collector import Collector, TensorCollector
from brainpy.base import io

math = Integrator = None

__all__ = [
  'Base',
]

logger = logging.getLogger('brainpy.base')


class Base(object):
  """The Base class for whole BrainPy ecosystem.

  The subclass of Base includes:

  - ``DynamicalSystem`` in *brainpy.simulation.brainobjects.base.py*
  - ``Integrator`` in *brainpy.integrators.base.py*
  - ``Function`` in *brainpy.base.function.py*
  - ``AutoGrad`` in *brainpy.math.jax.autograd.py*
  - ``Optimizer`` in *brainpy.math.jax.optimizers.py*
  - ``Scheduler`` in *brainpy.math.jax.optimizers.py*

  """
  implicit_vars = None
  """Used to wrap the implicit variables which cannot be accessed by self.xxx"""

  implicit_nodes = None
  """Used to wrap the implicit children nodes which cannot be accessed by self.xxx"""

  def __init__(self, name=None):
    # check whether the object has a unique name.
    self.name = self.unique_name(name=name)
    namechecking.check_name(name=self.name, obj=self)

  def vars(self, method='absolute'):
    """Collect all variables in this node and the children nodes.

    Parameters
    ----------
    method : str
      The method to access the variables.

    Returns
    -------
    gather : TensorCollector
      The collection contained (the path, the variable).
    """
    global math
    if math is None: from brainpy import math

    nodes = self.nodes(method=method)
    gather = TensorCollector()
    for node_path, node in nodes.items():
      for k in dir(node):
        v = getattr(node, k)
        if isinstance(v, math.Variable):
          gather[f'{node_path}.{k}' if node_path else k] = v
      if node.implicit_vars is not None:
        gather.update({f'{node_path}.{k}': v for k, v in node.implicit_vars.items()})
    return gather

  def train_vars(self, method='absolute'):
    """The shortcut for retrieving all trainable variables.

    Parameters
    ----------
    method : str
      The method to access the variables. Support 'absolute' and 'relative'.

    Returns
    -------
    gather : TensorCollector
      The collection contained (the path, the trainable variable).
    """
    global math
    if math is None:
      from brainpy import math
    return self.vars(method=method).subset(math.TrainVar)

  def nodes(self, method='absolute', _paths=None):
    """Collect all children nodes.

    Parameters
    ----------
    method : str
      The method to access the nodes.
    _paths : set, Optional
      The data structure to solve the circular reference.

    Returns
    -------
    gather : Collector
      The collection contained (the path, the node).
    """
    if _paths is None:
      _paths = set()
    gather = Collector()
    if method == 'absolute':
      nodes = []
      for k, v in self.__dict__.items():
        if isinstance(v, Base):
          path = (id(self), id(v))
          if path not in _paths:
            _paths.add(path)
            gather[v.name] = v
            nodes.append(v)
      if self.implicit_nodes is not None:
        assert isinstance(self.implicit_nodes, dict)
        for node in self.implicit_nodes.values():
          path = (id(self), id(node))
          if path not in _paths:
            _paths.add(path)
            gather[node.name] = node
            nodes.append(node)
      for v in nodes:
        gather.update(v.nodes(method=method, _paths=_paths))
      gather[self.name] = self

    elif method == 'relative':
      nodes = []
      gather[''] = self
      for k, v in self.__dict__.items():
        if isinstance(v, Base):
          path = (id(self), id(v))
          if path not in _paths:
            _paths.add(path)
            gather[k] = v
            nodes.append((k, v))
      if self.implicit_nodes is not None:
        assert isinstance(self.implicit_nodes, dict)
        for key, node in self.implicit_nodes.items():
          path = (id(self), id(node))
          if path not in _paths:
            _paths.add(path)
            gather[key] = node
            nodes.append((key, node))
      for k1, v1 in nodes:
        for k2, v2 in v1.nodes(method=method, _paths=_paths).items():
          if k2: gather[f'{k1}.{k2}'] = v2

    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def ints(self, method='absolute'):
    """Collect all integrators in this node and the children nodes.

    Parameters
    ----------
    method : str
      The method to access the integrators.

    Returns
    -------
    collector : Collector
      The collection contained (the path, the integrator).
    """
    global Integrator
    if Integrator is None: from brainpy.integrators.base import Integrator

    nodes = self.nodes(method=method)
    gather = Collector()
    for node_path, node in nodes.items():
      for k in dir(node):
        v = getattr(node, k)
        if isinstance(v, Integrator):
          gather[f'{node_path}.{k}' if node_path else k] = v
    return gather

  def unique_name(self, name=None, type=None):
    """Get the unique name for this object.

    Parameters
    ----------
    name : str, optional
      The expected name. If None, the default unique name will be returned.
      Otherwise, the provided name will be checked to guarantee its uniqueness.
    type : str, optional
      The type of this class, used for object naming.

    Returns
    -------
    name : str
      The unique name for this object.
    """
    if name is None:
      if type is None:
        return namechecking.get_name(type=self.__class__.__name__)
      else:
        return namechecking.get_name(type=type)
    else:
      namechecking.check_name(name=name, obj=self)
      return name

  def load_states(self, filename, verbose=False, check=False):
    """Load the model states.

    Parameters
    ----------
    filename : str
      The filename which stores the model states.
    """
    if not os.path.exists(filename):
      raise errors.BrainPyError(f'Cannot find the file path: {filename}')
    elif filename.endswith('.hdf5') or filename.endswith('.h5'):
      io.load_h5(filename, target=self, verbose=verbose, check=check)
    elif filename.endswith('.pkl'):
      io.load_pkl(filename, target=self, verbose=verbose, check=check)
    elif filename.endswith('.npz'):
      io.load_npz(filename, target=self, verbose=verbose, check=check)
    elif filename.endswith('.mat'):
      io.load_mat(filename, target=self, verbose=verbose, check=check)
    else:
      raise errors.BrainPyError(f'Unknown file format: {filename}. We only supports {io.SUPPORTED_FORMATS}')

  def save_states(self, filename, all_vars=None, **setting):
    """Save the model states.

    Parameters
    ----------
    filename : str
      The file name which to store the model states.
    """
    if all_vars is None:
      all_vars = self.vars(method='relative').unique()

    if filename.endswith('.hdf5') or filename.endswith('.h5'):
      io.save_h5(filename, all_vars=all_vars)
    elif filename.endswith('.pkl'):
      io.save_pkl(filename, all_vars=all_vars)
    elif filename.endswith('.npz'):
      io.save_npz(filename, all_vars=all_vars, **setting)
    elif filename.endswith('.mat'):
      io.save_mat(filename, all_vars=all_vars)
    else:
      raise errors.BrainPyError(f'Unknown file format: {filename}. We only supports {io.SUPPORTED_FORMATS}')

  def to(self, devices):
    global math
    if math is None: from brainpy import math

  def cpu(self):
    global math
    if math is None: from brainpy import math

    all_vars = self.vars().unique()
    for data in all_vars.values():
      data[:] = math.asarray(data.value)
      # TODO

  def cuda(self):
    global math
    if math is None: from brainpy import math

  def tpu(self):
    global math
    if math is None: from brainpy import math
