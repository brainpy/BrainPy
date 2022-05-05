# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Union, Callable

import tqdm.auto
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_map

from brainpy.base import Base
import brainpy.math as bm
from brainpy.errors import NoImplementationError
from brainpy.nn.algorithms.online import get, OnlineAlgorithm, RLS
from brainpy.nn.base import Node
from brainpy.nn.utils import (serialize_kwargs,
                              check_data_batch_size,
                              check_rnn_data_time_step)
from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'OnlineTrainer',
  'ForceTrainer',
]


class OnlineTrainer(RNNTrainer):
  """Online trainer for models with recurrent dynamics.

  Parameters
  ----------
  target: Node
    The target model to train.
  fit_method: OnlineAlgorithm, Callable, dict, str
    The fitting method applied to the target model.
    - It can be a string, which specify the shortcut name of the training algorithm.
      Like, ``fit_method='ridge'`` means using the RLS method.
      All supported fitting methods can be obtained through
      :py:func:`brainpy.nn.runners.get_supported_online_methods`
    - It can be a dict, whose "name" item specifies the name of the training algorithm,
      and the others parameters specify the initialization parameters of the algorithm.
      For example, ``fit_method={'name': 'ridge', 'beta': 1e-4}``.
    - It can be an instance of :py:class:`brainpy.nn.runners.OnlineAlgorithm`.
      For example, ``fit_meth=bp.nn.runners.RLS(alpha=1e-5)``.
    - It can also be a callable function.
  **kwargs
    The other general parameters for RNN running initialization.
  """

  def __init__(
      self,
      target: Node,
      fit_method: Union[OnlineAlgorithm, Callable, Dict, str] = None,
      **kwargs
  ):
    super(OnlineTrainer, self).__init__(target=target, **kwargs)

    # training method
    if fit_method is None:
      fit_method = RLS(alpha=1e-7)
    elif isinstance(fit_method, str):
      fit_method = get(fit_method)()
    elif isinstance(fit_method, dict):
      name = fit_method.pop('name')
      fit_method = get(name)(**fit_method)
    self.fit_method = fit_method
    if not callable(fit_method):
      raise ValueError(f'"train_method" must be an instance of callable function, '
                       f'but we got {type(fit_method)}.')

    # check the required interface in the trainable nodes
    self._check_interface()

    # set the training method
    for node in self.train_nodes:
      node.online_fit_by = fit_method

    # initialize the fitting method
    for node in self.train_nodes:
      node.online_init()

    # update dynamical variables
    if isinstance(self.fit_method, Base):
      self.dyn_vars.update(self.fit_method.vars().unique())

    # training function
    self._f_train = dict()

  def __repr__(self):
    name = self.__class__.__name__
    prefix = ' ' * len(name)
    return (f'{name}(target={self.target}, \n\t'
            f'{prefix}jit={self.jit}, \n\t'
            f'{prefix}fit_method={self.fit_method})')

  def fit(
      self,
      train_data: Sequence,
      test_data=None,
      reset: bool = False,
      shared_kwargs: Dict = None,
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None,
      initial_states: Dict[str, Tensor] = None,
      initial_feedbacks: Dict[str, Tensor] = None,
  ):
    # checking training and testing data
    if not isinstance(train_data, (list, tuple)):
      raise ValueError(f"{self.__class__.__name__} only support "
                       f"training data with the format of (X, Y) pair, "
                       f"but we got a {type(train_data)}.")
    if len(train_data) != 2:
      raise ValueError(f"{self.__class__.__name__} only support "
                       f"training data with the format of (X, Y) pair, "
                       f"but we got a sequence with length {len(train_data)}")
    if test_data is not None:
      raise ValueError(f'{self.__class__.__name__} does not support testing data.')
    xs, ys = train_data

    # format input data
    xs, num_step, num_batch = self._check_xs(xs, move_axis=True)

    # format target data
    ys = self._check_ys(ys, num_batch=num_batch, num_step=num_step, move_axis=True)

    # set initial states
    self._set_initial_states(initial_states)
    self._set_initial_feedbacks(initial_feedbacks)

    # get forced data
    forced_states = self._check_forced_states(forced_states, num_batch, num_step)
    forced_feedbacks = self._check_forced_feedbacks(forced_feedbacks, num_batch, num_step)

    # reset the model states
    if reset:
      self.target.initialize(num_batch)

    # init monitor
    for key in self.mon.item_contents.keys():
      self.mon.item_contents[key] = []  # reshape the monitor items

    # init progress bar
    if self.progress_bar:
      if num_step is None:
        num_step = check_rnn_data_time_step(xs)
      self._pbar = tqdm.auto.tqdm(total=num_step)
      self._pbar.set_description(f"Train {num_step} steps: ", refresh=True)

    # prediction
    hists = self._fit(xs=xs,
                      ys=ys,
                      forced_states=forced_states,
                      forced_feedbacks=forced_feedbacks,
                      shared_kwargs=shared_kwargs)

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # post-running for monitors
    for key in self.mon.item_names:
      self.mon.item_contents[key] = hists[key]
    if self.numpy_mon_after_run:
      self.mon.numpy()

  def _fit(
      self,
      xs: Dict[str, Tensor],
      ys: Dict[str, Tensor],
      shared_kwargs: Dict = None,
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None,
  ):
    """Predict the output according to the inputs.

    Parameters
    ----------
    xs: dict
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    ys: dict
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    forced_states: dict
      The forced state values.
    forced_feedbacks: dict
      The forced feedback output values.
    shared_kwargs: optional, dict
      The shared keyword arguments.

    Returns
    -------
    outputs, hists
      A tuple of pair of (outputs, hists).
    """
    _predict_func = self._get_fit_func(shared_kwargs)
    # rune the model
    forced_states = dict() if forced_states is None else forced_states
    forced_feedbacks = dict() if forced_feedbacks is None else forced_feedbacks
    hists = _predict_func([xs, ys, forced_states, forced_feedbacks])
    f1 = lambda x: bm.moveaxis(x, 0, 1)
    f2 = lambda x: isinstance(x, bm.JaxArray)
    hists = tree_map(f1, hists, is_leaf=f2)
    return hists

  def _get_fit_func(self, shared_kwargs: Dict = None):
    if shared_kwargs is None: shared_kwargs = dict()
    shared_kwargs_str = serialize_kwargs(shared_kwargs)
    if shared_kwargs_str not in self._f_train:
      self._f_train[shared_kwargs_str] = self._make_fit_func(shared_kwargs)
    return self._f_train[shared_kwargs_str]

  def _make_fit_func(self, shared_kwargs: Dict):
    if not isinstance(shared_kwargs, dict):
      raise ValueError(f'"shared_kwargs" must be a dict, '
                       f'but got {type(shared_kwargs)}')
    add_monitors = self._add_monitor_items()

    def _step_func(all_inputs):
      xs, ys, forced_states, forced_feedbacks = all_inputs
      monitors = tuple(self.mon.item_contents.keys())

      _, outs = self.target(xs,
                            forced_states=forced_states,
                            forced_feedbacks=forced_feedbacks,
                            monitors=monitors + add_monitors,
                            **shared_kwargs)
      for node in self.train_nodes:
        ff = outs[f'{node.name}.inputs']
        fb = outs[f'{node.name}.feedbacks']
        target = ys[node.name]
        node.online_fit(target, ff, fb=fb)
      for key in add_monitors:
        outs.pop(key)

      if self.progress_bar and (self._pbar is not None):
        id_tap(lambda *args: self._pbar.update(), ())
      return outs

    if self.jit['fit']:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      f = bm.make_loop(_step_func, dyn_vars=dyn_vars.unique(), has_return=True)
      return lambda all_inputs: f(all_inputs)[1]

    else:
      def run_func(all_inputs):
        xs, ys, forced_states, forced_feedbacks = all_inputs
        monitors = {key: [] for key in self.mon.item_contents.keys()}
        num_step = check_data_batch_size(xs)
        for i in range(num_step):
          one_xs = {key: tensor[i] for key, tensor in xs.items()}
          one_ys = {key: tensor[i] for key, tensor in ys.items()}
          one_forced_states = {key: tensor[i] for key, tensor in forced_states.items()}
          one_forced_feedbacks = {key: tensor[i] for key, tensor in forced_feedbacks.items()}
          mon = _step_func([one_xs, one_ys, one_forced_states, one_forced_feedbacks])
          for key, value in mon.items():
            monitors[key].append(value)
        for key, value in monitors.items():
          monitors[key] = bm.asarray(value)
        return monitors
    return run_func

  def _add_monitor_items(self):
    added_items = set()
    for node in self.train_nodes:
      if f'{node.name}.inputs' not in self.mon.item_names:
        added_items.add(f'{node.name}.inputs')
      if f'{node.name}.feedbacks' not in self.mon.item_names:
        added_items.add(f'{node.name}.feedbacks')
    return tuple(added_items)

  def _check_interface(self):
    for node in self.train_nodes:
      if hasattr(node.online_fit, 'not_implemented'):
        if node.online_fit.not_implemented:
          raise NoImplementationError(
            f'The node \n\n{node}\n\n'
            f'is set to be trainable with {self.__class__.__name__} method. '
            f'However, it does not implement the required training '
            f'interface "online_fit()" function. '
          )
      if hasattr(node.online_init, 'not_implemented'):
        if node.online_init.not_implemented:
          raise NoImplementationError(
            f'The node \n\n{node}\n\n'
            f'is set to be trainable with {self.__class__.__name__} method. '
            f'However, it does not implement the required training '
            f'interface "online_init()" function. '
          )


class ForceTrainer(OnlineTrainer):
  """Force learning."""

  def __init__(self, target, alpha=1., **kwargs):
    fit_method = RLS(alpha=alpha)
    super(ForceTrainer, self).__init__(target=target,
                                       fit_method=fit_method,
                                       **kwargs)
