# -*- coding: utf-8 -*-

import math as pm
import warnings
from typing import Union, Dict, Callable, Sequence, List, Optional

import jax.numpy as jnp
import numpy as np

import brainpy.math as bm
from brainpy import tools
from brainpy.base.base import Base
from brainpy.base.collector import Collector
from brainpy.connect import TwoEndConnector, MatConn, IJConn
from brainpy.errors import ModelBuildError
from brainpy.initialize import Initializer, init_param, Uniform
from brainpy.integrators import Integrator, odeint
from brainpy.tools.others import to_size, size2num
from brainpy.types import Tensor, Shape

__all__ = [
  'DynamicalSystem',
  'Container',
  'Network',
  'ConstantDelay',
  'NeuGroup',
  'ConNeuGroup',
  'TwoEndConn',
  'Channel',

  'ContainerWrapper',
]

_error_msg = 'Unknown type of the update function: {} ({}). ' \
             'Currently, BrainPy only supports: \n' \
             '1. function \n' \
             '2. function name (str) \n' \
             '3. tuple/dict of functions \n' \
             '4. tuple of function names \n'


class DynamicalSystem(Base):
  """Base Dynamical System class.

  Any object has step functions will be a dynamical system.
  That is to say, in BrainPy, the essence of the dynamical system
  is the "step functions".

  Parameters
  ----------
  name : str, optional
      The name of the dynamic system.
  """

  """Global delay variables. Useful when the same target
     variable is used in multiple mappings."""
  global_delay_vars: Dict[str, bm.LengthDelay] = Collector()
  global_delay_targets: Dict[str, bm.Variable] = Collector()

  def __init__(self, name=None):
    super(DynamicalSystem, self).__init__(name=name)

    # local delay variables
    self.local_delay_vars: Dict[str, bm.LengthDelay] = Collector()

  def __repr__(self):
    return f'{self.__class__.__name__}(name={self.name})'

  @property
  def steps(self):
    warnings.warn('.steps has been deprecated since version 2.0.3.', DeprecationWarning)
    return {}

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
    nodes = self.nodes(method=method)
    gather = Collector()
    for node_path, node in nodes.items():
      for k in dir(node):
        v = getattr(node, k)
        if isinstance(v, Integrator):
          gather[f'{node_path}.{k}' if node_path else k] = v
    return gather

  def __call__(self, *args, **kwargs):
    """The shortcut to call ``update`` methods."""
    return self.update(*args, **kwargs)

  def register_delay(
      self,
      name: str,
      delay_step: Optional[Union[int, Tensor, Callable, Initializer]],
      delay_target: bm.Variable,
      initial_delay_data: Union[Initializer, Callable, Tensor, float, int, bool] = None,
  ):
    """Register delay variable.

    Parameters
    ----------
    name: str
      The delay variable name.
    delay_step: Optional, int, JaxArray, ndarray, callable, Initializer
      The number of the steps of the delay.
    delay_target: Variable
      The target variable for delay.
    initial_delay_data: float, int, JaxArray, ndarray, callable, Initializer
      The initializer for the delay data.

    Returns
    -------
    delay_step: int, JaxArray, ndarray
      The number of the delay steps.
    """
    # delay steps
    if delay_step is None:
      return delay_step
    elif isinstance(delay_step, int):
      delay_type = 'homo'
    elif isinstance(delay_step, (bm.ndarray, jnp.ndarray, np.ndarray)):
      if delay_step.size == 1 and delay_step.ndim == 0:
        delay_type = 'homo'
      else:
        delay_type = 'heter'
        delay_step = bm.asarray(delay_step)
    elif callable(delay_step):
      delay_step = init_param(delay_step, delay_target.shape, allow_none=False)
      delay_type = 'heter'
    else:
      raise ValueError(f'Unknown "delay_steps" type {type(delay_step)}, only support '
                       f'integer, array of integers, callable function, brainpy.init.Initializer.')
    if delay_type == 'heter':
      if delay_step.dtype not in [bm.int32, bm.int64]:
        raise ValueError('Only support delay steps of int32, int64. If your '
                         'provide delay time length, please divide the "dt" '
                         'then provide us the number of delay steps.')
      if delay_target.shape[0] != delay_step.shape[0]:
        raise ValueError(f'Shape is mismatched: {delay_target.shape[0]} != {delay_step.shape[0]}')
    if delay_type != 'none':
      max_delay_step = int(bm.max(delay_step))

    # delay target
    if not isinstance(delay_target, bm.Variable):
      raise ValueError(f'"delay_target" must be an instance of Variable, but we got {type(delay_target)}')

    # delay variable
    self.global_delay_targets[name] = delay_target
    if delay_type != 'none':
      if name not in self.global_delay_vars:
        self.global_delay_vars[name] = bm.LengthDelay(delay_target, max_delay_step, initial_delay_data)
        self.local_delay_vars[name] = self.global_delay_vars[name]
      else:
        if self.global_delay_vars[name].num_delay_step - 1 < max_delay_step:
          self.global_delay_vars[name].reset(delay_target, max_delay_step, initial_delay_data)
    self.register_implicit_nodes(self.global_delay_vars)
    return delay_step

  def get_delay_data(
      self,
      name: str,
      delay_step: Optional[Union[int, bm.JaxArray, jnp.DeviceArray]],
      *indices: Union[int, bm.JaxArray, jnp.DeviceArray],
  ):
    """Get delay data according to the provided delay steps.

    Parameters
    ----------
    name: str
      The delay variable name.
    delay_step: Optional, int, JaxArray, ndarray
      The delay length.
    indices: optional, int, JaxArray, ndarray
      The indices of the delay.

    Returns
    -------
    delay_data: JaxArray, ndarray
      The delay data at the given time.
    """
    if delay_step is None:
      return self.global_delay_targets[name]

    if name in self.global_delay_vars:
      if isinstance(delay_step, int):
        return self.global_delay_vars[name](delay_step, *indices)
      else:
        if len(indices) == 0:
          indices = (jnp.arange(delay_step.size),)
        return self.global_delay_vars[name](delay_step, *indices)

    elif name in self.local_delay_vars:
      if isinstance(delay_step, int):
        return self.local_delay_vars[name](delay_step)
      else:
        if len(indices) == 0:
          indices = (jnp.arange(delay_step.size),)
        return self.local_delay_vars[name](delay_step, *indices)

    else:
      raise ValueError(f'{name} is not defined in delay variables.')

  def update_delay(
      self,
      name: str,
      delay_data: Union[float, bm.JaxArray, jnp.ndarray]
  ):
    """Update the delay according to the delay data.

    Parameters
    ----------
    name: str
      The name of the delay.
    delay_data: float, JaxArray, ndarray
      The delay data to update at the current time.
    """
    warnings.warn('All registered delays by "register_delay()" will be '
                  'automatically updated in the network model since 2.1.13. '
                  'Explicitly call "update_delay()" has no effect.',
                  DeprecationWarning)
    # if name in self.local_delay_vars:
    #   return self.local_delay_vars[name].update(delay_data)
    # else:
    #   if name not in self.global_delay_vars:
    #     raise ValueError(f'{name} is not defined in delay variables.')

  def reset_delay(
      self,
      name: str,
      delay_target: Union[bm.JaxArray, jnp.DeviceArray]
  ):
    """Reset the delay variable."""
    warnings.warn('All registered delays by "register_delay()" will be '
                  'automatically reset in the network model since 2.1.13. '
                  'Explicitly call "reset_delay()" has no effect.',
                  DeprecationWarning)
    # if name in self.local_delay_vars:
    #   return self.local_delay_vars[name].reset(delay_target)
    # else:
    #   if name not in self.global_delay_vars:
    #     raise ValueError(f'{name} is not defined in delay variables.')

  def update(self, t, dt):
    """The function to specify the updating rule.
    Assume any dynamical system depends on the time variable ``t`` and
    the time step ``dt``.
    """
    raise NotImplementedError('Must implement "update" function by subclass self.')

  def reset(self):
    """Reset function which reset the whole variables in the model.
    """
    raise NotImplementedError('Must implement "reset" function by subclass self.')


class Container(DynamicalSystem):
  """Container object which is designed to add other instances of DynamicalSystem.

  Parameters
  ----------
  steps : tuple of function, tuple of str, dict of (str, function), optional
      The step functions.
  monitors : tuple, list, Monitor, optional
      The monitor object.
  name : str, optional
      The object name.
  show_code : bool
      Whether show the formatted code.
  ds_dict : dict of (str, )
      The instance of DynamicalSystem with the format of "key=dynamic_system".
  """

  def __init__(self, *ds_tuple, name=None, **ds_dict):
    super(Container, self).__init__(name=name)

    # children dynamical systems
    self.implicit_nodes = Collector()
    for ds in ds_tuple:
      if not isinstance(ds, DynamicalSystem):
        raise ModelBuildError(f'{self.__class__.__name__} receives instances of '
                              f'DynamicalSystem, however, we got {type(ds)}.')
      if ds.name in self.implicit_nodes:
        raise ValueError(f'{ds.name} has been paired with {ds}. Please change a unique name.')
    self.register_implicit_nodes({node.name: node for node in ds_tuple})
    for key, ds in ds_dict.items():
      if not isinstance(ds, DynamicalSystem):
        raise ModelBuildError(f'{self.__class__.__name__} receives instances of '
                              f'DynamicalSystem, however, we got {type(ds)}.')
      if key in self.implicit_nodes:
        raise ValueError(f'{key} has been paired with {ds}. Please change a unique name.')
    self.register_implicit_nodes(ds_dict)

  def __repr__(self):
    cls_name = self.__class__.__name__
    split = ', '
    children = [f'{key}={str(val)}' for key, val in self.implicit_nodes.items()]
    return f'{cls_name}({split.join(children)})'

  def update(self, t, dt):
    """Update function of a container.

    In this update function, the update functions in children systems are
    iteratively called.
    """
    nodes = self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique()
    for node in nodes.values():
      node.update(t, dt)

  def __getitem__(self, item):
    """Wrap the slice access (self['']). """
    if item in self.implicit_nodes:
      return self.implicit_nodes[item]
    else:
      raise ValueError(f'Unknown item {item}, we only found {list(self.implicit_nodes.keys())}')

  def __getattr__(self, item):
    """Wrap the dot access ('self.'). """
    child_ds = super(Container, self).__getattribute__('implicit_nodes')
    if item in child_ds:
      return child_ds[item]
    else:
      return super(Container, self).__getattribute__(item)

  @classmethod
  def has(cls, **children_cls):
    """The aggressive operation to gather master and children classes.

    Parameters
    ----------
    children_cls
      The children classes.

    Returns
    -------
    wrapper: ContainerWrapper
      A wrapper which has master and its children classes.
    """
    return ContainerWrapper(master=cls, **children_cls)


class Network(Container):
  """Base class to model network objects, an alias of Container.

  Network instantiates a network, which is aimed to load
  neurons, synapses, and other brain objects.

  Parameters
  ----------
  name : str, Optional
    The network name.
  monitors : optional, list of str, tuple of str
    The items to monitor.
  ds_tuple :
    A list/tuple container of dynamical system.
  ds_dict :
    A dict container of dynamical system.
  """

  def __init__(self, *ds_tuple, name=None, **ds_dict):
    super(Network, self).__init__(*ds_tuple, name=name, **ds_dict)

  def update(self, t, dt):
    """Step function of a network.

    In this update function, the update functions in children systems are
    iteratively called.
    """
    nodes = self.nodes(level=1, include_self=False)
    nodes = nodes.subset(DynamicalSystem)
    nodes = nodes.unique()
    neuron_groups = nodes.subset(NeuGroup)
    synapse_groups = nodes.subset(TwoEndConn)
    other_nodes = nodes - neuron_groups - synapse_groups

    # reset synapse nodes
    for node in synapse_groups.values():
      node.update(t, dt)

    # reset neuron nodes
    for node in neuron_groups.values():
      node.update(t, dt)

    # reset other types of nodes
    for node in other_nodes.values():
      node.update(t, dt)

    # reset delays
    for node in nodes.values():
      for name in node.local_delay_vars.keys():
        self.global_delay_vars[name].update(self.global_delay_targets[name].value)

  def reset(self):
    nodes = self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique()
    neuron_groups = nodes.subset(NeuGroup)
    synapse_groups = nodes.subset(TwoEndConn)

    # reset neuron nodes
    for node in neuron_groups.values():
      node.reset()

    # reset synapse nodes
    for node in synapse_groups.values():
      node.reset()

    # reset other types of nodes
    for node in (nodes - neuron_groups - synapse_groups).values():
      node.reset()

    # reset delays
    for node in nodes:
      for name in node.local_delay_vars.keys():
        self.global_delay_vars[name].reset(self.global_delay_targets[name])


class ConstantDelay(DynamicalSystem):
  """Class used to model constant delay variables.

  This class automatically supports batch size on the last axis. For example, if
  you run batch with the size of (10, 100), where `100` are batch size, then this
  class can automatically support your batched data.
  For examples,

  >>> import brainpy as bp
  >>> bp.dyn.ConstantDelay(size=(10, 100), delay=10.)

  This class also support nonuniform delays.

  >>> bp.dyn.ConstantDelay(size=100, delay=bp.math.random.random(100) * 4 + 10)

  Parameters
  ----------
  size : int, list of int, tuple of int
    The delay data size.
  delay : int, float, function, ndarray
    The delay time. With the unit of `dt`.
  dt: float, optional
    The time precision.
  name : optional, str
    The name of the dynamic system.
  """

  def __init__(self, size, delay, dtype=None, dt=None, **kwargs):
    # dt
    self.dt = bm.get_dt() if dt is None else dt
    self.dtype = dtype

    # data size
    if isinstance(size, int): size = (size,)
    if not isinstance(size, (tuple, list)):
      raise ModelBuildError(f'"size" must a tuple/list of int, but we got {type(size)}: {size}')
    self.size = tuple(size)

    # delay time length
    self.delay = delay

    # data and operations
    if isinstance(delay, (int, float)):  # uniform delay
      self.uniform_delay = True
      self.num_step = int(pm.ceil(delay / self.dt)) + 1
      self.out_idx = bm.Variable(bm.array([0], dtype=bm.uint32))
      self.in_idx = bm.Variable(bm.array([self.num_step - 1], dtype=bm.uint32))
      self.data = bm.Variable(bm.zeros((self.num_step,) + self.size, dtype=dtype))
      self.num = 1

    else:  # non-uniform delay
      self.uniform_delay = False
      if not len(self.size) == 1:
        raise NotImplementedError(f'Currently, BrainPy only supports 1D heterogeneous '
                                  f'delays, while we got the heterogeneous delay with '
                                  f'{len(self.size)}-dimensions.')
      self.num = tools.size2num(size)
      if bm.ndim(delay) != 1:
        raise ModelBuildError(f'Only support a 1D non-uniform delay. '
                              f'But we got {delay.ndim}D: {delay}')
      if delay.shape[0] != self.size[0]:
        raise ModelBuildError(f"The first shape of the delay time size must "
                              f"be the same with the delay data size. But "
                              f"we got {delay.shape[0]} != {self.size[0]}")
      delay = bm.around(delay / self.dt)
      self.diag = bm.array(bm.arange(self.num))
      self.num_step = bm.array(delay, dtype=bm.uint32) + 1
      self.in_idx = bm.Variable(self.num_step - 1)
      self.out_idx = bm.Variable(bm.zeros(self.num, dtype=bm.uint32))
      self.data = bm.Variable(bm.zeros((self.num_step.max(),) + size, dtype=dtype))

    super(ConstantDelay, self).__init__(**kwargs)

  def reset(self):
    """Reset the variables."""
    self.in_idx[:] = self.num_step - 1
    self.out_idx[:] = 0
    self.data[:] = 0

  @property
  def oldest(self):
    return self.pull()

  @property
  def latest(self):
    if self.uniform_delay:
      return self.data[self.in_idx[0]]
    else:
      return self.data[self.in_idx, self.diag]

  def pull(self):
    if self.uniform_delay:
      return self.data[self.out_idx[0]]
    else:
      return self.data[self.out_idx, self.diag]

  def push(self, value):
    if self.uniform_delay:
      self.data[self.in_idx[0]] = value
    else:
      self.data[self.in_idx, self.diag] = value

  def update(self, t=None, dt=None, **kwargs):
    """Update the delay index."""
    self.in_idx[:] = (self.in_idx + 1) % self.num_step
    self.out_idx[:] = (self.out_idx + 1) % self.num_step


class NeuGroup(DynamicalSystem):
  """Base class to model neuronal groups.

  There are several essential attributes:

  - ``size``: the geometry of the neuron group. For example, `(10, )` denotes a line of
    neurons, `(10, 10)` denotes a neuron group aligned in a 2D space, `(10, 15, 4)` denotes
    a 3-dimensional neuron group.
  - ``num``: the flattened number of neurons in the group. For example, `size=(10, )` => \
    `num=10`, `size=(10, 10)` => `num=100`, `size=(10, 15, 4)` => `num=600`.

  Parameters
  ----------
  size : int, tuple of int, list of int
    The neuron group geometry.
  name : optional, str
    The name of the dynamic system.
  keep_size: bool
    Whether keep the geometry information.
  """

  def __init__(
      self,
      size: Shape,
      name: str = None,
      keep_size: bool = False
  ):
    # size
    if isinstance(size, (list, tuple)):
      if len(size) <= 0:
        raise ModelBuildError(f'size must be int, or a tuple/list of int. '
                              f'But we got {type(size)}')
      if not isinstance(size[0], int):
        raise ModelBuildError('size must be int, or a tuple/list of int.'
                              f'But we got {type(size)}')
      size = tuple(size)
    elif isinstance(size, int):
      size = (size,)
    else:
      raise ModelBuildError('size must be int, or a tuple/list of int.'
                            f'But we got {type(size)}')
    self.size = size
    self.keep_size = keep_size
    # number of neurons
    self.num = tools.size2num(size)
    self.var_shape = self.size if self.keep_size else self.num

    # initialize
    super(NeuGroup, self).__init__(name=name)

  def update(self, t, dt):
    """The function to specify the updating rule.

    Parameters
    ----------
    t : float
      The current time.
    dt : float
      The time step.
    """
    raise NotImplementedError(f'Subclass of {self.__class__.__name__} must '
                              f'implement "update" function.')


class TwoEndConn(DynamicalSystem):
  """Base class to model two-end synaptic connections.

  Parameters
  ----------
  pre : NeuGroup
      Pre-synaptic neuron group.
  post : NeuGroup
      Post-synaptic neuron group.
  conn : optional, ndarray, JaxArray, dict, TwoEndConnector
      The connection method between pre- and post-synaptic groups.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]] = None,
      name: str = None
  ):

    # pre or post neuron group
    # ------------------------
    if not isinstance(pre, NeuGroup):
      raise ModelBuildError('"pre" must be an instance of NeuGroup.')
    if not isinstance(post, NeuGroup):
      raise ModelBuildError('"post" must be an instance of NeuGroup.')
    self.pre = pre
    self.post = post

    # connectivity
    # ------------
    if isinstance(conn, TwoEndConnector):
      self.conn = conn(pre.size, post.size)
    elif isinstance(conn, (bm.ndarray, np.ndarray, jnp.ndarray)):
      if (pre.num, post.num) != conn.shape:
        raise ModelBuildError(f'"conn" is provided as a matrix, and it is expected '
                              f'to be an array with shape of (pre.num, post.num) = '
                              f'{(pre.num, post.num)}, however we got {conn.shape}')
      self.conn = MatConn(conn_mat=conn)
    elif isinstance(conn, dict):
      if not ('i' in conn and 'j' in conn):
        raise ModelBuildError(f'"conn" is provided as a dict, and it is expected to '
                              f'be a dictionary with "i" and "j" specification, '
                              f'however we got {conn}')
      self.conn = IJConn(i=conn['i'], j=conn['j'])
    elif isinstance(conn, str):
      self.conn = conn
    elif conn is None:
      self.conn = None
    else:
      raise ModelBuildError(f'Unknown "conn" type: {conn}')

    # initialize
    # ----------
    super(TwoEndConn, self).__init__(name=name)

  def check_pre_attrs(self, *attrs):
    """Check whether pre group satisfies the requirement."""
    if not hasattr(self, 'pre'):
      raise ModelBuildError('Please call __init__ function first.')
    for attr in attrs:
      if not isinstance(attr, str):
        raise ValueError(f'Must be string. But got {attr}.')
      if not hasattr(self.pre, attr):
        raise ModelBuildError(f'{self} need "pre" neuron group has attribute "{attr}".')

  def check_post_attrs(self, *attrs):
    """Check whether post group satisfies the requirement."""
    if not hasattr(self, 'post'):
      raise ModelBuildError('Please call __init__ function first.')
    for attr in attrs:
      if not isinstance(attr, str):
        raise ValueError(f'Must be string. But got {attr}.')
      if not hasattr(self.post, attr):
        raise ModelBuildError(f'{self} need "pre" neuron group has attribute "{attr}".')


class ConNeuGroup(NeuGroup, Container):
  """Base class to model conductance-based neuron group.

  The standard formulation for a conductance-based model is given as

  .. math::

      C_m {dV \over dt} = \sum_jg_j(E - V) + I_{ext}

  where :math:`g_j=\bar{g}_{j} M^x N^y` is the channel conductance, :math:`E` is the
  reversal potential, :math:`M` is the activation variable, and :math:`N` is the
  inactivation variable.

  :math:`M` and :math:`N` have the dynamics of

  .. math::

      {dx \over dt} = \phi_x {x_\infty (V) - x \over \tau_x(V)}

  where :math:`x \in [M, N]`, :math:`\phi_x` is a temperature-dependent factor,
  :math:`x_\infty` is the steady state, and :math:`\tau_x` is the time constant.
  Equivalently, the above equation can be written as:

  .. math::

      \frac{d x}{d t}=\phi_{x}\left(\alpha_{x}(1-x)-\beta_{x} x\right)

  where :math:`\alpha_{x}` and :math:`\beta_{x}` are rate constants.

  .. versionadded:: 2.1.9

  Parameters
  ----------
  size : int, sequence of int
    The network size of this neuron group.
  method: str
    The numerical integration method.
  name : optional, str
    The neuron group name.

  """

  def __init__(
      self,
      size: Shape,
      keep_size:bool=False,
      C: Union[float, Tensor, Initializer, Callable] = 1.,
      A: Union[float, Tensor, Initializer, Callable] = 1e-3,
      V_th: Union[float, Tensor, Initializer, Callable] = 0.,
      V_initializer: Union[Initializer, Callable, Tensor] = Uniform(-70, -60.),
      method: str = 'exp_auto',
      name: str = None,
      **channels
  ):
    NeuGroup.__init__(self, size, keep_size=keep_size)
    Container.__init__(self, **channels, name=name)

    # parameters for neurons
    self.C = C
    self.A = A
    self.V_th = V_th
    self._V_initializer = V_initializer

    # variables
    self.V = bm.Variable(init_param(V_initializer, self.var_shape, allow_none=False))
    self.input = bm.Variable(bm.zeros(self.var_shape))
    self.spike = bm.Variable(bm.zeros(self.var_shape, dtype=bool))

    # function
    self.integral = odeint(self.derivative, method=method)

  def reset(self):
    self.V.value = init_param(self._V_initializer, self.var_shape, allow_none=False)
    self.spike[:] = False
    self.input[:] = 0

  def derivative(self, V, t):
    Iext = self.input.value * (1e-3 / self.A)
    for ch in self.implicit_nodes.unique().values():
      Iext = Iext + ch.current(V)
    return Iext / self.C

  def update(self, t, dt):
    V = self.integral(self.V.value, t, dt)
    for node in self.implicit_nodes.unique().values():
      node.update(t, dt, self.V.value)
    self.spike.value = bm.logical_and(V >= self.V_th, self.V < self.V_th)
    self.input[:] = 0.
    self.V.value = V


class Channel(DynamicalSystem):
  """Abstract channel model."""

  master_type = ConNeuGroup

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      name: str = None,
      keep_size: bool = False,
  ):
    super(Channel, self).__init__(name=name)
    # the geometry size
    self.size = to_size(size)
    # the number of elements
    self.num = size2num(self.size)
    # variable shape
    self.keep_size = keep_size
    self.var_shape = self.size if self.keep_size else self.num

  def update(self, t, dt):
    raise NotImplementedError('Must be implemented by the subclass.')

  def current(self):
    raise NotImplementedError('Must be implemented by the subclass.')

  def reset(self):
    raise NotImplementedError('Must be implemented by the subclass.')


class ContainerWrapper(object):
  def __init__(self, master, **children):
    self.master = master
    self.children_cls = children

    if not isinstance(master, type):
      raise TypeError(f'"master" should be a type. But we got {master}')
    for key, child in children.items():
      if isinstance(child, type):
        if not issubclass(child, Channel):
          raise TypeError(f'{child} should be a subclass of Base.')
        if child.master_type is None:
          raise TypeError(f'{child} should set its master_type.')
        if not issubclass(master, child.master_type):
          raise TypeError(f'Type does not match. {child} requires a master with type '
                          f'of {child.master_type}, but the master now is {master}.')
      elif isinstance(child, ContainerWrapper):
        if not issubclass(child.master, Channel):
          raise TypeError(f'{child.master} should be a subclass of Base.')
        if child.master.master_type is None:
          raise TypeError(f'{child.master} should set its master_type.')
        if not issubclass(master, child.master.master_type):
          raise TypeError(f'Type does not match. {child.master} requires a master with type '
                          f'of {child.master.master_type}, but the master now is {master}.')

      else:
        raise TypeError(f'The item in children should be a type or '
                        f'{ContainerWrapper.__name__} instance. But we got {child}')

  def __call__(self, size, *shared_args, shared_kwargs=None, **idv_args):
    if shared_kwargs is None:
      shared_kwargs = dict()

    # initialize children classes
    children = dict()
    for key, cls in self.children_cls.items():
      if key in idv_args:
        pars = idv_args.pop(key)
      else:
        pars = dict()
      children[key] = cls(size, *shared_args, **shared_kwargs, **pars)

    # initialize master class
    master = self.master(size, *shared_args, **shared_kwargs, **idv_args, **children)

    # assign master or parent to children
    for child in children.values():
      child.master = master

    return master

  def __repr__(self):
    children = [f'{key}={val.__name__}' for key, val in self.children_cls.items()]
    return f'{self.master.__name__}({", ".join(children)})'
