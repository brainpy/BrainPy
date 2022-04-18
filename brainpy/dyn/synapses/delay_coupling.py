# -*- coding: utf-8 -*-

from typing import Optional, Union, Sequence, Dict, List

from jax import vmap

import brainpy.math as bm
from brainpy.dyn.base import TwoEndConn, NeuGroup
from brainpy.initialize import Initializer, ZeroInit
from brainpy.tools.checking import check_sequence
from brainpy.types import Tensor

__all__ = [
  'DelayCoupling',
  'DiffusiveDelayCoupling',
  'AdditiveDelayCoupling',
]


class DelayCoupling(TwoEndConn):
  """
  Delay coupling base class.

  Parameters
  ----------
  coupling: str
    The way of coupling.
  gc: float
    The global coupling strength.
  signal_speed: float
    Signal transmission speed between areas.
  sc_mat: optional, tensor
    Structural connectivity matrix. Adjacency matrix of coupling strengths,
    will be normalized to 1. If not given, then a single node simulation
    will be assumed. Default None
  fl_mat: optional, tensor
    Fiber length matrix. Will be used for computing the
    delay matrix together with the signal transmission
    speed parameter `signal_speed`. Default None.

  """



  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      from_to: Union[str, Sequence[str]],
      conn_mat: Tensor,
      delay_mat: Optional[Tensor] = None,
      delay_initializer: Initializer = ZeroInit(),
      domain: str = 'local',
      name: str = None
  ):
    super(DelayCoupling, self).__init__(pre, post, name=name)

    # domain
    if domain not in ['global', 'local']:
      raise ValueError('"domain" must be a string in ["global", "local"]. '
                       f'Bug we got {domain}.')
    self.domain = domain

    # pairs of (source, destination)
    self.source_target_pairs: Dict[str, List[bm.Variable]] = dict()
    source_vars = {}
    if isinstance(from_to, str):
      from_to = [from_to]
    check_sequence(from_to, 'from_to', elem_type=str, allow_none=False)
    for pair in from_to:
      splits = [v.strip() for v in pair.split('->')]
      if len(splits) != 2:
        raise ValueError('The (source, target) pair in "from_to" '
                         'should be defined as "a -> b".')
      if not hasattr(self.pre, splits[0]):
        raise ValueError(f'"{splits[0]}" is not defined in pre-synaptic group {self.pre.name}')
      if not hasattr(self.post, splits[1]):
        raise ValueError(f'"{splits[1]}" is not defined in post-synaptic group {self.post.name}')
      source = f'{self.pre.name}.{splits[0]}'
      target = getattr(self.post, splits[1])
      if splits[0] not in self.source_target_pairs:
        self.source_target_pairs[source] = [target]
        source_vars[source] = getattr(self.pre, splits[0])
        if not isinstance(source_vars[source], bm.Variable):
          raise ValueError(f'The target variable {source} for delay should '
                           f'be an instance of brainpy.math.Variable, while '
                           f'we got {type(source_vars[source])}')
      else:
        if target in self.source_target_pairs:
          raise ValueError(f'{pair} has been defined twice in {from_to}.')
        self.source_target_pairs[source].append(target)

    # Connection matrix
    conn_mat = bm.asarray(conn_mat)
    required_shape = (self.post.num, self.pre.num)
    if conn_mat.shape != required_shape:
      raise ValueError(f'we expect the structural connection matrix has the shape of '
                       f'(post.num, pre.num), i.e., {required_shape}, '
                       f'while we got {conn_mat.shape}.')
    self.conn_mat = bm.asarray(conn_mat)
    bm.fill_diagonal(self.conn_mat, 0)

    # Delay matrix
    if delay_mat is None:
      self.delay_mat = bm.zeros(required_shape, dtype=bm.int_)
    else:
      if delay_mat.shape != required_shape:
        raise ValueError(f'we expect the fiber length matrix has the shape of '
                         f'(post.num, pre.num), i.e., {required_shape}. '
                         f'While we got {delay_mat.shape}.')
      self.delay_mat = bm.asarray(delay_mat, dtype=bm.int_)

    # delay variables
    num_delay_step = int(self.delay_mat.max())
    for var in self.source_target_pairs.keys():
      if domain == 'local':
        variable = source_vars[var]
        shape = (num_delay_step,) + variable.shape
        delay_data = delay_initializer(shape, dtype=variable.dtype)
        self.local_delay_vars[var] = bm.LengthDelay(variable, num_delay_step, delay_data)
      else:
        if var not in self.global_delay_vars:
          variable = source_vars[var]
          shape = (num_delay_step,) + variable.shape
          delay_data = delay_initializer(shape, dtype=variable.dtype)
          self.global_delay_vars[var] = bm.LengthDelay(variable, num_delay_step, delay_data)
          # save into local delay vars when first seen "var",
          # for later update current value!
          self.local_delay_vars[var] = self.global_delay_vars[var]
        else:
          if self.global_delay_vars[var].num_delay_step - 1 < num_delay_step:
            variable = source_vars[var]
            shape = (num_delay_step,) + variable.shape
            delay_data = delay_initializer(shape, dtype=variable.dtype)
            self.global_delay_vars[var].init(variable, num_delay_step, delay_data)

    self.register_implicit_nodes(self.local_delay_vars)
    self.register_implicit_nodes(self.global_delay_vars)

  def update(self, _t, _dt):
    raise NotImplementedError('Must implement the update() function by users.')


class DiffusiveDelayCoupling(DelayCoupling):
  def update(self, _t, _dt):
    for source, targets in self.source_target_pairs.items():
      # delay variable
      if self.domain == 'local':
        delay_var: bm.LengthDelay = self.local_delay_vars[source]
      elif self.domain == 'global':
        delay_var: bm.LengthDelay = self.global_delay_vars[source]
      else:
        raise ValueError(f'Unknown domain: {self.domain}')

      # current data
      name, var = source.split('.')
      assert name == self.pre.name
      variable = getattr(self.pre, var)

      # delays
      f = vmap(lambda i: delay_var(self.delay_mat[i], bm.arange(self.pre.num)))  # (pre.num,)
      delays = f(bm.arange(self.post.num).value)
      diffusive = delays - bm.expand_dims(variable, axis=1)  # (post.num, pre.num)
      diffusive = (self.conn_mat * diffusive).sum(axis=1)

      # output to target variable
      for target in targets:
        target.value += diffusive

      # update
      if source in self.local_delay_vars:
        delay_var.update(variable)


class AdditiveDelayCoupling(DelayCoupling):
  def update(self, _t, _dt):
    for source, targets in self.source_target_pairs.items():
      # delay variable
      if self.domain == 'local':
        delay_var: bm.LengthDelay = self.local_delay_vars[source]
      elif self.domain == 'global':
        delay_var: bm.LengthDelay = self.global_delay_vars[source]
      else:
        raise ValueError(f'Unknown domain: {self.domain}')

      # current data
      name, var = source.split('.')
      assert name == self.pre.name
      variable = getattr(self.pre, var)

      # delay function
      f = vmap(lambda i: delay_var(self.delay_mat[i], bm.arange(self.pre.num)))  # (pre.num,)
      delays = f(bm.arange(self.post.num))  # (post.num, pre.num)
      additive = (self.conn_mat * delays).sum(axis=1)

      # output to target variable
      for target in targets:
        target.value += additive

      # update
      if source in self.local_delay_vars:
        delay_var.update(variable)
