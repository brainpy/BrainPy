# -*- coding: utf-8 -*-

import logging
import os.path

from brainpy import errors, tools
from brainpy.tools import namechecking, collector

math = DE_INT = None

__all__ = [
  'Base',
]

logger = logging.getLogger('brainpy.base')


class Base(object):
  """The Base class for whole BrainPy ecosystem.

  The subclass of Base includes:

  - ``Module`` in brainpy.dnn.base.py
  - ``DynamicalSystem`` in brainpy.simulation.brainobjects.base.py

  """
  target_backend = None

  def __init__(self, name=None):
    # check whether the object has a unique name.
    self.name = self.unique_name(name=name)
    namechecking.check_name(name=self.name, obj=self)

    # target backend
    if self.target_backend is None:
      self.target_backend = ('general',)
    elif isinstance(self.target_backend, str):
      self.target_backend = (self.target_backend,)
    elif isinstance(self.target_backend, (tuple, list)):
      if not isinstance(self.target_backend[0], str):
        raise errors.BrainPyError('"target_backend" must be a list/tuple of string.')
      self.target_backend = tuple(self.target_backend)
    else:
      raise errors.BrainPyError(f'Unknown setting of "target_backend": {self.target_backend}')

    # check target backend
    global math
    if math is None: from brainpy import math
    check1 = self.target_backend[0] != 'general'
    check2 = math.get_backend_name() not in self.target_backend
    if check1 and check2:
      msg = f'ERROR: The model {self.name} is target to run on {self.target_backend}, ' \
            f'but currently the selected backend is "{math.get_backend_name()}"'
      logger.error(msg)
      raise errors.BrainPyError(msg)

  def vars(self, method='absolute'):
    """Collect all variables in this node and the children nodes.

    Parameters
    ----------
    method : str
      The method to access the variables.

    Returns
    -------
    gather : collector.ArrayCollector
      The collection contained (the path, the variable).
    """
    global math
    if math is None: from brainpy import math

    nodes = self.nodes(method=method)
    gather = collector.ArrayCollector()
    for node_path, node in nodes.items():
      for k in dir(node):
        v = getattr(node, k)
        if isinstance(v, math.Variable):
          gather[f'{node_path}.{k}' if node_path else k] = v
    return gather

  def train_vars(self, method='absolute'):
    """The shortcut for retrieving all trainable variables.

    Parameters
    ----------
    method : str
      The method to access the variables. Support 'absolute' and 'relative'.

    Returns
    -------
    gather : collector.ArrayCollector
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
    gather : collector.Collector
      The collection contained (the path, the node).
    """
    if _paths is None:
      _paths = set()
    gather = collector.Collector()
    if method == 'absolute':
      nodes = []
      for k, v in self.__dict__.items():
        if isinstance(v, Base):
          path = (id(self), id(v))
          if path not in _paths:
            _paths.add(path)
            gather[v.name] = v
            nodes.append(v)
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
      for k, v in nodes:
        for k2, v2 in v.nodes(method=method, _paths=_paths).items():
          if k2:
            gather[f'{k}.{k2}'] = v2

    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def _nodes_in_container(self, dict_container, method='absolute', _paths=None):
    if _paths is None:
      _paths = set()

    gather = collector.Collector()
    if method == 'absolute':
      nodes = []
      for _, node in dict_container.items():
        path = (id(self), id(node))
        if path not in _paths:
          _paths.add(path)
          gather[node.name] = node
          nodes.append(node)
      for node in nodes:
        gather[node.name] = node
        gather.update(node.nodes(method=method, _paths=_paths))
      gather[self.name] = self

    elif method == 'relative':
      nodes = []
      gather[''] = self
      for key, node in dict_container.items():
        path = (id(self), id(node))
        if path not in _paths:
          _paths.add(path)
          gather[key] = node
          nodes.append((key, node))
      for key, node in nodes:
        for key2, node2 in node.nodes(method=method, _paths=_paths).items():
          if key2: gather[f'{key}.{key2}'] = node2

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
    collector : collector.Collector
      The collection contained (the path, the integrator).
    """
    global DE_INT
    if DE_INT is None:
      from brainpy.integrators.constants import DE_INT

    nodes = self.nodes(method=method)
    gather = collector.Collector()
    for node_path, node in nodes.items():
      for k in dir(node):
        v = getattr(node, k)
        if callable(v) and hasattr(v, '__name__') and v.__name__.startswith(DE_INT):
          gather[f'{node_path}.{k}' if node_path else k] = v
    return gather

  def unique_name(self, name=None, type=None):
    """Get the unique name for this object.

    Parameters
    ----------
    name : str, optional
      The expected name. If None, the default unique name will be returned.
      Otherwise, the provided name will be checked to guarantee its
      uniqueness.
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

  def load_states(self, filename):
    """Load the model states.

    Parameters
    ----------
    filename : str
      The filename which stores the model states.
    """
    if not os.path.exists(filename):
      raise errors.BrainPyError(f'Cannot find the file path: {filename}')
    if filename.endswith('.hdf5') or filename.endswith('.h5'):
      tools.io.load_h5(filename, target=self)
    if filename.endswith('.pkl'):
      tools.io.load_pkl(filename, target=self)
    if filename.endswith('.npz'):
      tools.io.load_npz(filename, target=self)
    if filename.endswith('.mat'):
      tools.io.load_mat(filename, target=self)
    raise errors.BrainPyError(f'Unknown file format: {filename}. We only supports {tools.io.SUPPORTED_FORMATS}')

  def save_states(self, filename, **setting):
    """Save the model states.

    Parameters
    ----------
    filename : str
      The file name which to store the model states.
    """
    if filename.endswith('.hdf5') or filename.endswith('.h5'):
      tools.io.save_h5(filename, all_vars=self.vars())
    if filename.endswith('.pkl'):
      tools.io.save_pkl(filename, all_vars=self.vars())
    if filename.endswith('.npz'):
      tools.io.save_npz(filename, all_vars=self.vars(), **setting)
    if filename.endswith('.mat'):
      tools.io.save_mat(filename, all_vars=self.vars())
    raise errors.BrainPyError(f'Unknown file format: {filename}. We only supports {tools.io.SUPPORTED_FORMATS}')

  def to(self, devices):
    global math
    if math is None: from brainpy import math

  def cpu(self):
    global math
    if math is None: from brainpy import math

    if math.get_backend_name() == 'jax':
      all_vars = self.vars().unique()
      for data in all_vars.values():
        data[:] = math.asarray(data.value)
        # TODO

  def cuda(self):
    global math
    if math is None: from brainpy import math
    if math.get_backend_name() != 'jax':
      raise errors.BrainPyError(f'Only support to deploy data into "tpu" device in "jax" backend. '
                                f'While currently the selected backend is "{math.get_backend_name()}".')

  def tpu(self):
    global math
    if math is None: from brainpy import math
    if math.get_backend_name() != 'jax':
      raise errors.BrainPyError(f'Only support to deploy data into "tpu" device in "jax" backend. '
                                f'While currently the selected backend is "{math.get_backend_name()}".')
