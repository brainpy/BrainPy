# -*- coding: utf-8 -*-

from typing import Sequence, List, Dict, Iterable

from brainpy.rnn import utils, graph_flow
from brainpy.rnn.base import Network


__all__ = [
  'Dispatcher',
]


class Dispatcher(object):
  _inputs: Sequence
  _parents: Dict

  def __init__(self, model):
    assert isinstance(model, Network)

    self._nodes = model.lnodes
    # self._trainables = model.trainable_nodes
    self._inputs = model.entry_nodes
    self.__parents, _ = graph_flow.find_parents_and_children(model.ff_edges)
    self._parents = utils.safe_defaultdict_copy(self.__parents)
    self._teachers = dict()

  def __getitem__(self, name):
    return self.get(name)

  def get(self, name):
    parents = self._parents.get(name, ())
    teacher = self._teachers.get(name, None)

    x = []
    for parent in parents:
      if is_node(parent):
        x.append(parent.state())
      else:
        x.append(parent)

    # in theory, only operators can support several incoming signal
    # i.e. several operands, so unpack data is the list is unnecessary
    if len(x) == 1:
      x = x[0]

    return DataPoint(x=x, y=teacher)

  def _check_inputs(self, input_mapping):
    if is_mapping(input_mapping):
      for node in self._inputs:
        if input_mapping.get(node.name) is None:
          raise KeyError(f"Node {node.name} not found "
                         f"in data mapping. This node requires "
                         f"data to run.")

  def _check_targets(self, target_mapping):
    if is_mapping(target_mapping):
      for node in self._nodes:
        if (
            node in self._trainables
            and not node.fitted
            and target_mapping.get(node.name) is None
        ):
          raise KeyError(
            f"Trainable node {node.name} not found "
            f"in target/feedback data mapping. This "
            f"node requires "
            f"target values."
          )

  def _format_xy(self, X, Y):
    if not is_mapping(X):
      X_map = {inp.name: X for inp in self._inputs}
    else:
      X_map = X.copy()
    self._check_inputs(X_map)

    Y_map = None
    if Y is not None:
      if not is_mapping(Y):
        Y_map = {trainable.name: Y for trainable in self._trainables}
      else:
        Y_map = Y.copy()
      self._check_targets(Y_map)

    # check if all sequences have same length,
    # taking the length of the first input sequence
    # as reference
    current_node = list(X_map.keys())[0]
    sequence_length = len(X_map[current_node])
    for node, sequence in X_map.items():
      if sequence_length != len(sequence):
        raise ValueError(
          f"Impossible to use data with inconsistent "
          f"number of timesteps: {node} is given "
          f"a sequence of length {len(sequence)} as "
          f"input while {current_node} is given a sequence "
          f"of length {sequence_length}"
        )

    if Y_map is not None:
      # Pad teacher nodes in a sequence
      for node, value in Y_map.items():
        if isinstance(value, Node):
          Y_map[node] = [value for _ in range(sequence_length)]

      for node, sequence in Y_map.items():
        # Y_map might be a teacher node (not a sequence)
        if hasattr(Y_map, "__len__"):
          if sequence_length != len(sequence):
            raise ValueError(
              f"Impossible to use data with inconsistent "
              f"number of timesteps: {node} is given "
              f"a sequence of length {len(sequence)} as "
              f"targets/feedbacks while {current_node} is "
              f"given a sequence of length {sequence_length}."
            )

    return X_map, Y_map, sequence_length

  def load(self, X=None, Y=None):
    self._parents = utils.safe_defaultdict_copy(self.__parents)
    self._teachers = dict()

    if X is not None:
      self._check_inputs(X)
      if is_mapping(X):
        for node in self._nodes:
          if X.get(node.name) is not None:
            self._parents[node] += [X[node.name]]

      else:
        for inp_node in self._inputs:
          self._parents[inp_node] += [X]

    if Y is not None:
      self._check_targets(Y)
      for node in self._nodes:
        if is_mapping(Y):
          if Y.get(node.name) is not None:
            self._teachers[node] = Y.get(node.name)
        else:
          if node in self._trainables:
            self._teachers[node] = Y
    return self

  def dispatch(
      self,
      X,
      Y=None,
      shift_fb=True,
      return_targets=False,
      force_teachers=True,
      format_xy=True,
  ):
    if format_xy:
      X_map, Y_map, sequence_length = self._format_xy(X, Y)
    else:
      X_map, Y_map = X, Y
      current_node = list(X_map.keys())[0]
      sequence_length = len(X_map[current_node])

    for i in range(sequence_length):
      x = {node: X_map[node][i] for node in X_map.keys()}
      if Y_map is not None:
        y = None
        if return_targets:
          y = {node: Y_map[node][i] for node in Y_map.keys()}
        # if feedbacks vectors are meant to be fed
        # with a delay in time of one timestep w.r.t. 'X_map'
        if shift_fb:
          if i == 0:
            if force_teachers:
              fb = {
                node: np.zeros_like(Y_map[node][i])
                for node in Y_map.keys()
              }
            else:
              fb = {node: None for node in Y_map.keys()}
          else:
            fb = {node: Y_map[node][i - 1] for node in Y_map.keys()}
        # else assume that all feedback vectors must be instantaneously
        # fed to the network. This means that 'Y_map' already contains
        # data that is delayed by one timestep w.r.t. 'X_map'.
        else:
          fb = {node: Y_map[node][i] for node in Y_map.keys()}
      else:
        fb = y = None

      yield x, fb, y
