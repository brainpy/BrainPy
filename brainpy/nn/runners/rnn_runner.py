# -*- coding: utf-8 -*-

from typing import Dict, Union

import jax.numpy as jnp
import tqdm.auto
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_map

from brainpy import math as bm
from brainpy.errors import UnsupportedError
from brainpy.nn.base import Node, Network
from brainpy.nn.utils import (check_rnn_data_time_step,
                              check_rnn_data_batch_size,
                              serialize_kwargs)
from brainpy.running.runner import Runner
from brainpy.tools.checking import check_dict_data
from brainpy.types import Tensor

__all__ = [
  'RNNRunner',
]


class RNNRunner(Runner):
  """Structural Runner for Recurrent Neural Networks.

  Parameters
  ----------
  target: Node
    The target model for simulation.
  monitors: None, list of str, tuple of str, Monitor
    Variables to monitor.
  jit: bool
    Whether use JIT compilation to accelerate the model simulation.
  progress_bar: bool
    Whether use progress bar to report the simulation progress.
  dyn_vars: Optional, dict
    The dynamically changed variables.
  numpy_mon_after_run : bool
    Change the monitored iterm into NumPy arrays.
  """

  def __init__(self, target: Node, **kwargs):
    super(RNNRunner, self).__init__(target=target, **kwargs)
    assert isinstance(self.target, Node), '"target" must be an instance of brainpy.nn.Node.'

    # function for prediction
    self._predict_func = dict()

  def predict(self,
              xs: Union[Tensor, Dict[str, Tensor]],
              forced_states: Dict[str, Tensor] = None,
              forced_feedbacks: Dict[str, Tensor] = None,
              reset=False,
              shared_kwargs: Dict = None,
              progress_bar=True):
    """Predict a series of input data with the given target model.

    This function use the JIT compilation to accelerate the model simulation.
    Moreover, it can automatically monitor the node variables, states, inputs,
    feedbacks and its output.

    Parameters
    ----------
    xs: Tensor, dict
      The feedforward input data. It must be a 3-dimensional data
      which has the shape of `(num_sample, num_time, num_feature)`.
    forced_states: dict
      The fixed node states. Similar with ``xs``, each tensor in
      ``forced_states`` must be a tensor with the shape of
      `(num_sample, num_time, num_feature)`.
    forced_feedbacks: dict
      The fixed feedback states. Similar with ``xs``, each tensor in
      ``forced_states`` must be a tensor with the shape of
      `(num_sample, num_time, num_feature)`.
    reset: bool
      Whether reset the model states.
    progress_bar: bool
    shared_kwargs: optional, dict
      The shared arguments across different layers.

    Returns
    -------
    output: Tensor, dict
      The model output.
    """
    # format input data
    xs, num_step, num_batch = self._check_xs(xs)
    # get forced data
    iter_forced_states, fixed_forced_states = \
      self._check_forced_states(forced_states, num_step, num_batch)
    iter_forced_feedbacks, fixed_forced_feedbacks = \
      self._check_forced_feedbacks(forced_feedbacks, num_step, num_batch)
    # reset the model states
    if reset:
      self.target.initialize(num_batch)
    # init monitor
    for key in self.mon.item_contents.keys():
      self.mon.item_contents[key] = []  # reshape the monitor items
    # init progress bar
    if self.progress_bar and progress_bar:
      if num_step is None:
        num_step = check_rnn_data_time_step(xs)
      self._pbar = tqdm.auto.tqdm(total=num_step)
      self._pbar.set_description(f"Predict {num_step} steps: ", refresh=True)
    # prediction
    outputs, hists = self._predict(xs=xs,
                                   iter_forced_states=iter_forced_states,
                                   iter_forced_feedbacks=iter_forced_feedbacks,
                                   shared_kwargs=shared_kwargs)
    # close the progress bar
    if self.progress_bar and progress_bar:
      self._pbar.close()
    # post-running for monitors
    for key in self.mon.item_names:
      self.mon.item_contents[key] = hists[key]
    if self.numpy_mon_after_run:
      self.mon.numpy()
    return outputs

  def _predict(
      self,
      xs: Dict[str, Tensor],
      iter_forced_states: Dict[str, Tensor] = None,
      iter_forced_feedbacks: Dict[str, Tensor] = None,
      shared_kwargs: Dict = None,
  ):
    """Predict the output according to the inputs.

    Parameters
    ----------
    xs: dict
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    iter_forced_states: dict
    iter_forced_feedbacks: dict
    shared_kwargs: dict

    Returns
    -------
    outputs, hists
      A tuple of pair of (outputs, hists).
    """
    _predict_func = self._get_predict_func(shared_kwargs)
    # rune the model
    iter_forced_states = dict() if iter_forced_states is None else iter_forced_states
    iter_forced_feedbacks = dict() if iter_forced_feedbacks is None else iter_forced_feedbacks
    outputs, hists = _predict_func([xs, iter_forced_states, iter_forced_feedbacks])  # TODO: fixed?
    f1 = lambda x: bm.moveaxis(x, 0, 1)
    f2 = lambda x: isinstance(x, bm.JaxArray)
    outputs = tree_map(f1, outputs, is_leaf=f2)
    hists = tree_map(f1, hists, is_leaf=f2)
    return outputs, hists

  def _get_predict_func(self, shared_kwargs: Dict = None):
    shared_kwargs_str = serialize_kwargs(shared_kwargs)
    if shared_kwargs_str not in self._predict_func:
      self._predict_func[shared_kwargs_str] = self._make_run_func(shared_kwargs)
    return self._predict_func[shared_kwargs_str]

  def _make_run_func(self, shared_kwargs: Dict = None):
    if shared_kwargs is None:
      shared_kwargs = dict()
    assert isinstance(shared_kwargs, dict), (f'"shared_kwargs" must be a dict, '
                                             f'but got {type(shared_kwargs)}')

    def _step_func(a_input):
      xs, forced_states, forced_feedbacks = a_input
      monitors = self.mon.item_contents.keys()
      outs = self.target(xs,
                         forced_states=forced_states,
                         forced_feedbacks=forced_feedbacks,
                         monitors=monitors,
                         **shared_kwargs)
      if self.progress_bar and (self._pbar is not None):
        id_tap(lambda *args: self._pbar.update(), ())
      return outs

    if self.jit:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      f = bm.make_loop(_step_func, dyn_vars=dyn_vars.unique(), has_return=True)
      return lambda all_inputs: f(all_inputs)[1]

    else:
      def run_func(all_inputs):
        xs, forced_states, forced_feedbacks = all_inputs
        if isinstance(self.target, Network) and len(self.target.exit_nodes) > 1:
          outputs = {node.name: [] for node in self.target.exit_nodes}
          output_type = 'network'
        else:
          outputs = []
          output_type = 'node'
        monitors = {key: [] for key in self.mon.item_contents.keys()}
        num_step = check_rnn_data_batch_size(xs)
        for i in range(num_step):
          one_xs = {key: tensor[i] for key, tensor in xs.items()}
          one_forced_states = {key: tensor[i] for key, tensor in forced_states.items()}
          one_forced_feedbacks = {key: tensor[i] for key, tensor in forced_feedbacks.items()}
          output, mon = _step_func([one_xs, one_forced_states, one_forced_feedbacks])
          for key, value in mon.items():
            monitors[key].append(value)
          if output_type == 'node':
            outputs.append(output)
          else:
            for key, out in output.items():
              outputs[key].append(out)
        if output_type == 'node':
          outputs = bm.asarray(outputs)
        else:
          for key, out in outputs.items():
            outputs[key] = bm.asarray(out)
        for key, value in monitors.items():
          monitors[key] = bm.asarray(value)
        return outputs, monitors
    return run_func

  def _init_target(self, xs):
    # we need to initialize the node or the network
    x = dict()
    for key, tensor in xs.items():
      assert isinstance(key, str), ('"xs" must a dict of (str, tensor), while we got '
                                    f'({type(key)}, {type(tensor)})')
      assert isinstance(tensor, (bm.ndarray, jnp.ndarray)), ('"xs" must a dict of (str, tensor), while we got '
                                                             f'({type(key)}, {type(tensor)})')
      x[key] = tensor[0]
    self.target.initialize(x)

  def _set_initial_states(self, initial_states):
    # initial states
    if initial_states is not None:
      if isinstance(self.target, Network):
        assert isinstance(initial_states, dict), ('"initial_states" must be a dict when the '
                                                  'target model is a brainpy.nn.Network instance. '
                                                  f'But we got {type(initial_states)}')
        for key, tensor in initial_states.items():
          assert isinstance(key, str), (f'"initial_states" must be a dict of (str, tensor). '
                                        f'But got a dict of ({type(key)}, {type(tensor)})')
          assert key in self.target, (f'Node "{key}" is not in the target model. '
                                      f'We only detect: \n{self.target.lnodes}')
          assert self.target[key].state is not None, (f'The target model {key} has no state. '
                                                      f'We cannot set its initial state.')
          self.target[key].state.value = tensor
      elif isinstance(self.target, Node):
        assert self.target.state is not None, (f'The target model {self.target.name} has no state. '
                                               f'We cannot set its initial state.')
        assert isinstance(initial_states, (jnp.ndarray, bm.ndarray)), ('"initial_states" must be a tensor, '
                                                                       f'but we got a {type(initial_states)}')

  def _set_initial_feedbacks(self, initial_feedbacks):  # TODO
    # initial feedback states
    if initial_feedbacks is not None:
      if isinstance(self.target, Network):
        assert isinstance(initial_feedbacks, dict), ('"initial_feedbacks" must be a dict when the '
                                                     'target model is a brainpy.nn.Network instance. '
                                                     f'But we got {type(initial_feedbacks)}')
        for key, tensor in initial_feedbacks.items():
          assert isinstance(key, str), (f'"initial_feedbacks" must be a dict of (str, tensor). '
                                        f'But got a dict of ({type(key)}, {type(tensor)})')
          assert key in self.target.feedback_states, (f'Node "{key}" is not in the feedback nodes of the '
                                                      f'target model. We only detect: \n'
                                                      f'{self.target.feedback_states.keys()}')
          self.target.feedback_states[key].value = tensor
      elif isinstance(self.target, Node):
        raise UnsupportedError('Do not support feedback in a single instance of brainpy.nn.Node.')

  def _check_forced_states(self, forced_states, num_step, num_batch):
    iter_forced_states = dict()
    fixed_forced_states = dict()
    if forced_states is not None:
      if isinstance(self.target, Network):
        assert isinstance(forced_states, dict), '"forced_states" must be a dict of (str, Tensor)'
        for key, tensor in forced_states.items():
          assert isinstance(key, str), (f'"forced_states" must be a dict of (str, tensor). '
                                        f'But got a dict of ({type(key)}, {type(tensor)})')
          assert key not in self.target, f'{self.target} does no have node {key}.'
          assert isinstance(tensor, (bm.ndarray, jnp.ndarray)), ('"forced_states" must a dict of (str, tensor), '
                                                                 'while we got ({type(key)}, {type(tensor)})')
          assert tensor.shape[0] == num_batch, (f'The number of the batch size ({tensor.shape[0]}) '
                                                f'of the forced state of {key} does not '
                                                f'match with the batch size in inputs {num_batch}.')
          if bm.ndim(tensor) == 2 and self.target[key].output_shape[1:] == tensor.shape[1:]:
            fixed_forced_states[key] = tensor
          elif bm.ndim(tensor) == 3 and self.target[key].output_shape[1:] == tensor.shape[2:]:
            assert tensor.shape[1] == num_step, (f'The number of the time step ({tensor.shape[1]}) '
                                                 f'of the forced state of {key} does not '
                                                 f'match with the time step in inputs {num_step}.')
            iter_forced_states[key] = tensor  # shape of (num_time, num_sample, num_feature)
          else:
            raise UnsupportedError(f'The forced state of {key} has the shape of '
                                   f'{tensor.shape}, which is not consistent with '
                                   f'its output shape {self.target[key].output_shape}. '
                                   f'Each tensor in forced state should have the shape '
                                   f'of (num_sample, num_time, num_feature) or '
                                   f'(num_sample, num_feature).')
      else:
        raise UnsupportedError('We do not support forced feedback state '
                               'for a single brainpy.nn.Node instance')
    return iter_forced_states, fixed_forced_states

  def _check_forced_feedbacks(self, forced_feedbacks, num_step, num_batch):
    iter_forced_feedbacks = dict()
    fixed_forced_feedbacks = dict()
    if forced_feedbacks is not None:
      if isinstance(self.target, Network):
        assert isinstance(forced_feedbacks, dict), '"forced_feedbacks" must be a dict of (str, Tensor)'
        feedback_node_names = [node.name for node in self.target.feedback_nodes]
        for key, tensor in forced_feedbacks.items():
          assert isinstance(key, str), (f'"forced_feedbacks" must be a dict of (str, tensor). '
                                        f'But got a dict of ({type(key)}, {type(tensor)})')
          assert key not in feedback_node_names, (f'{self.target} has no feedback node {key}, '
                                                  f'it only has {feedback_node_names}')
          assert isinstance(tensor, (bm.ndarray, jnp.ndarray)), ('"forced_feedbacks" must a dict of (str, tensor), '
                                                                 'while we got ({type(key)}, {type(tensor)})')
          assert tensor.shape[0] == num_batch, (f'The number of the batch size ({tensor.shape[0]}) '
                                                f'of the forced feedback of {key} does not '
                                                f'match with the batch size in inputs {num_batch}.')
          if bm.ndim(tensor) == 2 and self.target[key].output_shape[1:] == tensor.shape[1:]:
            fixed_forced_feedbacks[key] = tensor
          elif bm.ndim(tensor) == 3 and self.target[key].output_shape[1:] == tensor.shape[2:]:
            assert tensor.shape[1] == num_step, (f'The number of the time step ({tensor.shape[1]}) '
                                                 f'of the forced feedback of {key} does not '
                                                 f'match with the time step in inputs {num_step}.')
            iter_forced_feedbacks[key] = bm.moveaxis(tensor, 0, 1)  # shape of (num_time, num_sample, num_feature)
          else:
            raise UnsupportedError(f'The forced feedback of {key} has the shape of '
                                   f'{tensor.shape}, which is not consistent with '
                                   f'its output shape {self.target[key].output_shape}. '
                                   f'Each tensor in forced feedback should have the shape '
                                   f'of (num_sample, num_time, num_feature) or '
                                   f'(num_sample, num_feature).')
      else:
        raise UnsupportedError('We do not support forced states for '
                               'a single brainpy.nn.Node instance')
    return iter_forced_feedbacks, fixed_forced_feedbacks

  def _format_xs(self, xs):
    if isinstance(xs, (bm.ndarray, jnp.ndarray)):
      if isinstance(self.target, Network):
        assert len(self.target.entry_nodes) == 1, (f'The network {self.target} has {len(self.target.entry_nodes)} '
                                                   f'input nodes, while we only got one input data.')
        xs = {self.target.entry_nodes[0].name: xs}
      else:
        xs = {self.target.name: xs}

    if not isinstance(xs, dict):
      raise UnsupportedError(f'Unknown data type {type(xs)}, we only support '
                             f'tensor or dict with <str, tensor>')
    assert len(xs) > 0, 'We got no input data.'
    check_dict_data(xs, key_type=str, val_type=(bm.ndarray, jnp.ndarray))
    return xs

  def _check_xs(self, xs: Union[Dict, Tensor], move_axis=True):
    xs = self._format_xs(xs)
    num_times, num_batch_sizes = [], []
    for key, val in xs.items():
      if bm.ndim(val) != 3:
        raise ValueError(f'Each tensor in "xs" must be a tensor of shape '
                         f'(num_sample, num_time, num_feature). But we got {val.shape}.')
      num_times.append(val.shape[1])
      num_batch_sizes.append(val.shape[0])
    if len(set(num_times)) != 1:
      raise ValueError(f'Number of time step is different across tensors in '
                       f'the provided "xs". We got {set(num_times)}.')
    if len(set(num_batch_sizes)) != 1:
      raise ValueError(f'Number of batch size is different across tensors in '
                       f'the provided "xs". We got {set(num_batch_sizes)}.')
    num_step = num_times[0]
    num_batch = num_batch_sizes[0]
    if move_axis:
      # change shape to (num_time, num_sample, num_feature)
      xs = {k: bm.moveaxis(v, 0, 1) for k, v in xs.items()}
    return xs, num_step, num_batch
