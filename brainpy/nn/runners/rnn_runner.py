# -*- coding: utf-8 -*-

from typing import Dict, Union

import jax.numpy as jnp
import tqdm.auto
from jax.experimental.host_callback import id_tap

from brainpy import math as bm
from brainpy.errors import UnsupportedError
from brainpy.nn.base import Node, Network
from brainpy.running.runner import Runner
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
    self._predict_func = None

  def predict(self,
              xs: Union[Tensor, Dict[str, Tensor]],
              forced_states: Dict[str, Tensor] = None,
              forced_feedbacks: Dict[str, Tensor] = None,
              initial_states: Dict[str, Tensor] = None,
              initial_feedbacks: Dict[str, Tensor] = None,
              reset=False):
    """Predict a series of input data with the given target model.

    This function use the JIT compilation to accelerate the model simulation.
    Moreover, it can automatically monitor the node variables, states, inputs,
    feedbacks and its output.

    Parameters
    ----------
    xs: Tensor, dict
      The feedforward input data.
    forced_states: dict
      The fixed node states.
    forced_feedbacks: dict
      The fixed feedback states.
    initial_states: dict
      The initial states for the target nodes.
    initial_feedbacks: dict
      The initial values of the feedback nodes.
    reset: bool
      Whether reset the model states.

    Returns
    -------
    output: Tensor, dict
      The model output.
    """
    # format input data
    xs, num_step = self._format_xs(xs)
    # init the target model
    self._init_target(xs)
    # init monitor
    for key in self.mon.item_contents.keys():
      self.mon.item_contents[key] = []  # reshape the monitor items
    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=num_step)
      self._pbar.set_description(f"Running {num_step} steps: ", refresh=True)
    # reset the model states
    if reset:
      self.target.reset_state()
    # set initial states/feedbacks
    self._set_initial_states(initial_states)
    self._set_initial_feedbacks(initial_feedbacks)
    # get forced data
    forced_states = self._check_forced_states(forced_states, num_step)
    forced_feedbacks = self._check_forced_feedbacks(forced_feedbacks, num_step)
    # check run function
    if self._predict_func is None:
      self._predict_func = self._make_run_func()
    # rune the model
    outputs, hists = self._predict_func([xs, forced_states, forced_feedbacks])
    # close the progress bar
    if self.progress_bar:
      self._pbar.close()
    # post-running for monitors
    for key in self.mon.item_names:
      self.mon.item_contents[key] = bm.asarray(hists[key])
    if self.numpy_mon_after_run: self.mon.numpy()
    return outputs

  def _step_func(self, a_input):
    xs, forced_states, forced_feedbacks = a_input
    monitors = self.mon.item_contents.keys()
    outs = self.target(xs,
                       forced_states=forced_states,
                       forced_feedbacks=forced_feedbacks,
                       monitors=monitors)
    if self.progress_bar:
      id_tap(lambda *args: self._pbar.update(), ())
    return outs

  def _make_run_func(self):
    if self.jit:
      self.dyn_vars.update(self.target.vars().unique())
      f = bm.make_loop(self._step_func, dyn_vars=self.dyn_vars, has_return=True)
      return lambda all_inputs: f(all_inputs)[1]

    else:
      def run_func(all_inputs):
        xs, forced_states, forced_feedbacks = all_inputs
        if isinstance(self.target, Network) and len(self.target.exit_nodes) > 1:
          outputs = {node.name: [] for node in self.target.exit_nodes}
          output_type = 1
        else:
          outputs = []
          output_type = 2
        monitors = {key: [] for key in self.mon.item_contents.keys()}
        num_step = list(xs.values())[0].shape[0]
        for i in range(num_step):
          one_xs = {key: tensor[i] for key, tensor in xs.items()}
          one_forced_states = {key: tensor[i] for key, tensor in forced_states.items()}
          one_forced_feedbacks = {key: tensor[i] for key, tensor in forced_feedbacks.items()}
          output, mon = self._step_func([one_xs, one_forced_states, one_forced_feedbacks])
          for key, value in mon.items():
            monitors[key].append(value)
          if output_type == 2:
            outputs.append(output)
          else:
            for key, out in output.items():
              outputs[key].append(out)
        if output_type == 2:
          outputs = bm.asarray(outputs)
        else:
          for key, out in outputs.items():
            outputs[key] = bm.asarray(out)
        for key, value in monitors.items():
          monitors[key] = bm.asarray(value)
        return outputs, monitors
    return run_func

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
    for key, tensor in xs.items():
      assert isinstance(key, str), ('"xs" must a dict of (str, tensor), while we got '
                                    f'({type(key)}, {type(tensor)})')
      assert isinstance(tensor, (bm.ndarray, jnp.ndarray)), ('"xs" must a dict of (str, tensor), while we got '
                                                             f'({type(key)}, {type(tensor)})')
    num_step = list(xs.values())[0].shape[0]
    return xs, num_step

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

  def _set_initial_feedbacks(self, initial_feedbacks):
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

  def _check_forced_states(self, forced_states, num_step):
    if forced_states is not None:
      if isinstance(self.target, Network):
        iter_forced_states = dict()
        assert isinstance(forced_states, dict), '"forced_states" must be a dict of (str, Tensor)'
        for key, tensor in forced_states.items():
          assert isinstance(key, str), (f'"forced_states" must be a dict of (str, tensor). '
                                        f'But got a dict of ({type(key)}, {type(tensor)})')
          assert key not in self.target, f'{self.target} does no have node {key}.'
          assert isinstance(tensor, (bm.ndarray, jnp.ndarray)), ('"forced_states" must a dict of (str, tensor), '
                                                                 'while we got ({type(key)}, {type(tensor)})')
          if self.target[key].output_shape == tensor.shape:
            iter_forced_states[key] = bm.repeat(bm.reshape(tensor, (-1,) + tensor.shape),
                                                num_step, axis=0)
          elif self.target[key].output_shape == tensor.shape[1:]:
            iter_forced_states[key] = tensor
            assert tensor.shape[0] == num_step, (f'The number of the iteration step ({tensor.shape[0]}) '
                                                 f'of the forced state of {key} does not '
                                                 f'match with the input data ({num_step}).')
          else:
            raise UnsupportedError(f'The forced state of {key} has the shape of '
                                   f'{tensor.shape}, which is not consistent with '
                                   f'its output shape {self.target[key].output_shape}.')
        return iter_forced_states
      else:
        raise UnsupportedError('We do not support forced feedback state '
                               'for a single brainpy.nn.Node instance')
    else:
      return dict()

  def _check_forced_feedbacks(self, forced_feedbacks, num_step):
    if forced_feedbacks is not None:
      if isinstance(self.target, Network):
        iter_forced_feedbacks = dict()
        assert isinstance(forced_feedbacks, dict), '"forced_feedbacks" must be a dict of (str, Tensor)'
        feedback_node_names = [node.name for node in self.target.feedback_nodes]
        for key, tensor in forced_feedbacks.items():
          assert isinstance(key, str), (f'"forced_feedbacks" must be a dict of (str, tensor). '
                                        f'But got a dict of ({type(key)}, {type(tensor)})')
          assert key not in feedback_node_names, (f'{self.target} has no feedback node {key}, '
                                                  f'it only has {feedback_node_names}')
          assert isinstance(tensor, (bm.ndarray, jnp.ndarray)), ('"forced_feedbacks" must a dict of (str, tensor), '
                                                                 'while we got ({type(key)}, {type(tensor)})')
          if self.target[key].output_shape == tensor.shape:
            iter_forced_feedbacks[key] = bm.repeat(bm.reshape(tensor, (-1,) + tensor.shape),
                                                   num_step, axis=0)
          elif self.target[key].output_shape == tensor.shape[1:]:
            iter_forced_feedbacks[key] = tensor
            assert tensor.shape[0] == num_step, (f'The number of the iteration step ({tensor.shape[0]}) '
                                                 f'of the forced feedback of {key} does not '
                                                 f'match with the input data ({num_step}).')
          else:
            raise UnsupportedError(f'The forced state of {key} has the shape of '
                                   f'{tensor.shape}, which is not consistent with '
                                   f'its output shape {self.target[key].output_shape}.')
        return iter_forced_feedbacks
      else:
        raise UnsupportedError('We do not support forced states for '
                               'a single brainpy.nn.Node instance')
    else:
      return dict()
