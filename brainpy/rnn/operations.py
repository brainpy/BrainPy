# -*- coding: utf-8 -*-


from itertools import product
from typing import Union, Sequence

from . import utils
from .base import Node, Network, FrozenNetwork
from .nodes import Concat, Select

__all__ = [
  'ff_connect', 'fb_connect', 'merge', 'select',
]


def _connect(node1: Node, node2: Node, name=None) -> Network:
  """Connects two nodes or networks."""
  # fetch all nodes in two subgraphs, if they are models.
  all_nodes = set()
  for node in (node1, node2):
    if isinstance(node, FrozenNetwork):
      raise TypeError(f"Cannot connect {FrozenNetwork.__name__} to other Nodes.")
    if isinstance(node, Network):
      all_nodes.update(set(node.lnodes))
    else:
      all_nodes.add(node)

  # fetch all feedforward edges in two subgraphs, if they are models.
  all_ff_edges = set()
  for node in (node1, node2):
    if isinstance(node, FrozenNetwork):
      raise TypeError(f"Cannot connect {FrozenNetwork.__name__} to other Nodes.")
    if isinstance(node, Network):
      all_ff_edges.update(set(node.ff_edges))

  # fetch all feedforward edges in two subgraphs, if they are models.
  all_fb_edges = set()
  for node in (node1, node2):
    if isinstance(node, FrozenNetwork):
      raise TypeError(f"Cannot connect {FrozenNetwork.__name__} to other Nodes.")
    if isinstance(node, Network):
      all_fb_edges.update(set(node.fb_edges))

  # create edges between output nodes of the
  # subgraph 1 and input nodes of the subgraph 2.
  senders = []
  if isinstance(node1, Network) and not isinstance(node, FrozenNetwork):
    senders += node1.exit_nodes
  else:
    senders.append(node1)
  receivers = []
  if isinstance(node2, Network) and not isinstance(node, FrozenNetwork):
    receivers += node2.entry_nodes
  else:
    receivers.append(node2)
  new_ff_edges = set(product(senders, receivers))

  # maybe nodes are already initialized ?
  # check if connected dimensions are ok
  for sender, receiver in new_ff_edges:
    if sender.is_initialized and receiver.is_initialized and (sender.out_size != receiver.in_size):
      raise ValueError(f"Dimension mismatch between connected nodes: "
                       f"sender node {sender.name} has output dimension "
                       f"{sender.out_size} but receiver node "
                       f"{receiver.name} has input dimension "
                       f"{receiver.in_size}.")

  # all outputs from subgraph 1 are connected to
  # all inputs from subgraph 2.
  all_ff_edges |= new_ff_edges

  # pack everything
  return Network(nodes=tuple(all_nodes),
                 ff_edges=tuple(all_ff_edges),
                 fb_edges=tuple(all_fb_edges),
                 name=name)


def ff_connect(
    sender: Union[Node, Sequence[Node]],
    receiver: Union[Node, Sequence[Node]],
    inplace: bool = False,
    name: str = None,
) -> Network:
  """Link two :py:class:`~.Node` instances to form a :py:class:`~.Network`
  instance. `node1` output will be used as input for `node2` in the
  created model. This is similar to a function composition operation:

  .. math::

      model(x) = (node1 \\circ node2)(x) = node2(node1(x))

  You can also perform this operation using the ``>>`` operator::

      model = node1 >> node2

  Or using this function::

      model = link(node1, node2)

  -`node1` and `node2` can also be :py:class:`~.Network` instances. In this
  case, the new :py:class:`~.Network` created will contain all nodes previously
  contained in all the models, and link all `node1` outputs to all `node2`
  inputs. This allows to chain the  ``>>`` operator::

      step1 = node0 >> node1  # this is a model
      step2 = step1 >> node2  # this is another

  -`node1` and `node2` can finally be lists or tuples of nodes. In this
  case, all `node1` outputs will be linked to a :py:class:`~.Concat` node to
  concatenate them, and the :py:class:`~.Concat` node will be linked to all
  `node2` inputs. You can still use the ``>>`` operator in this situation,
  except for many-to-many nodes connections::

      # many-to-one
      model = [node1, node2, ..., node] >> node_out
      # one-to-many
      model = node_in >> [node1, node2, ..., node]
      # ! many-to-many requires to use the `link` method explicitely !
      model = link([node1, node2, ..., node], [node1, node2, ..., node])

  Parameters
  ----------
  sender, receiver : Node or sequence of Node
      Nodes or lists of nodes to link.
  inplace: bool
  name: str, optional
      Name for the chaining Network.

  Returns
  -------
  Network
      A :py:class:`~.Network` instance chaining the nodes.

  Raises
  ------
  TypeError
      Dimension mismatch between connected nodes: `node1` output
      dimension if different from `node2` input dimension.
      Reinitialize the nodes or create new ones.

  Notes
  -----

  Be careful to how you link the different nodes: `reservoirpy` does not
  allow to have circular dependencies between them::

      model = node1 >> node2  # fine
      model = node1 >> node2 >> node1  # raises! data would flow in
                                       # circles forever...
  """
  # checking
  utils.check_all_nodes(sender, receiver)
  frozen_nets = []
  if isinstance(sender, Sequence):
    frozen_nets += [n for n in sender if isinstance(n, FrozenNetwork)]
  else:
    if isinstance(sender, FrozenNetwork):
      frozen_nets.append(sender)
  if isinstance(receiver, Sequence):
    frozen_nets += [n for n in receiver if isinstance(n, FrozenNetwork)]
  else:
    if isinstance(receiver, FrozenNetwork):
      frozen_nets.append(receiver)
  if len(frozen_nets) > 0:
    raise TypeError(f"Impossible to link {FrozenNetwork.__name__} to other Nodes or "
                    f"Network. {FrozenNetwork.__name__} found: {frozen_nets}.")

  # get left side
  if isinstance(sender, Sequence):
    if inplace:
      raise ValueError(f'Cannot inplace connect a sequence of node {sender} '
                       f'with other nodes. Must be a {Network.__name__}.')
    left_model = Network()
    if isinstance(receiver, Concat):  # no need to add a Concat node then
      for n in sender:
        left_model = merge(left_model, _connect(n, receiver), inplace=True)
      return left_model
    else:  # concatenate everything on left side
      concat = Concat()
      for n in sender:
        left_model = merge(left_model, _connect(n, concat), inplace=True)
  else:
    left_model = sender

  # connect left side with right side
  if isinstance(receiver, Sequence):
    model = Network(name=name)
    for n in receiver:
      model = merge(model, _connect(left_model, n), inplace=True)
  else:
    if inplace:
      if not isinstance(sender, Network):
        raise ValueError(f'Cannot inplace connect a node {sender} '
                         f'with other nodes. Must be a {Network.__name__}.')
      if name is not None:
        raise ValueError('Cannot set name when inplace=True.')
      model = merge(left_model, _connect(left_model, receiver), inplace=True)
    else:
      model = Network(name=name)
      model = merge(model, _connect(left_model, receiver), inplace=True)
  return model


def fb_connect(
    receiver: Union[Node, Sequence[Node]],
    sender: Union[Node, Sequence[Node]],
    inplace: bool = False,
    name: str = None,
) -> Node:
  """Create a feedback connection between the `feedback` node and `node`.
  Feedbacks nodes will be called at runtime using data from the previous
  call.

  This is not an inplace operation by default. This function will copy `node`
  and then sets the copy `_feedback` attribute as a reference to `feedback`
  node. If `inplace` is set to `True`, then `node` is not copied and the
  feedback is directly connected to `node`. If `feedback` is a list of nodes
  or models, then all nodes in the list are first connected to a
  :py:class:`~.Concat` node to create a model gathering all data from all nodes
  in a single feedback vector.

   You can also perform this operation using the ``<<`` operator::

      node1 = node1 << node2
      # with feedback from a Model
      node1 = node1 << (fbnode1 >> fbnode2)
      # with feedback from a list of nodes or models
      node1 = node1 << [fbnode1, fbnode2, ...]
      # with feedback from a list of nodes or models
      node1 = [node1, ...] << [fbnode1, fbnode2, ...]

  Which means that a feedback connection is now created between `node1` and
  `node2`. In other words, the forward function of `node1` depends on the
  previous output of `node2`:

  .. math::
      \\mathrm{node1}(x_t) = \\mathrm{node1}(x_t, \\mathrm{node2}(x_{t - 1}))

  You can also use this function to define feedback::

      node1 = link_feedback(node1, node2)
      # without copy (node1 is the same object throughout)
      node1 = link_feedback(node1, node2, inplace=True, name="n1_copy")

  Parameters
  ----------
  receiver : Node
      Node receiving feedback.
  sender : GenericNode
      Node or Model sending feedback
  inplace : bool, defaults to False
      If `True`, then the function returns a copy of `node`.
  name : str, optional
      Name of the copy of `node` if `inplace` is `True`.

  Returns
  -------
  Node
      A node instance taking feedback from `feedback`.

  Raises
  ------
  TypeError
      - If `node` is a :py:class:`~.Model`.
      Models can not receive feedback.

      - If any of the feedback nodes are not :py:class:`~.GenericNode`
      instances.
  """

  # checking
  if isinstance(receiver, Network):
    raise TypeError(f"{receiver} is not a Node. {Network.__name__} instance can't receive feedback.")
  if isinstance(sender, Sequence):
    for fb in sender:
      if not isinstance(fb, Node):
        raise TypeError(f"Impossible to receive feedback from {fb}: it is not "
                        f"a {Node.__name__} or a {Network.__name__} instance.")
    feedback = ff_connect(sender, Concat())
  elif isinstance(sender, Node):
    feedback = sender
  else:
    raise TypeError(f"Impossible to receive feedback from {sender}: it is not "
                    f"a {Node.__name__} or a {Network.__name__} instance.")

  # feedback
  if inplace:
    if receiver.has_feedback:
      receiver.feedback = feedback & receiver.feedback
    else:
      receiver.feedback = feedback
    return receiver
  else:
    # first copy the node, then give it feedback
    # original node is not connected to any feedback then
    new_node = receiver.copy(name=name)
    if new_node.has_feedback:
      new_node.feedback = feedback & new_node.feedback
    else:
      new_node.feedback = feedback
    return new_node


def merge(
    node: Node,
    *other_nodes: Node,
    inplace: bool = False,
    name: str = None
) -> Network:
  """Merge different :py:class:`~.Model` or :py:class:`~.Node`
  instances into a single :py:class:`~.Model` instance.

  :py:class:`~.Node` instances contained in the models to merge will be
  gathered in a single model, along with all previously defined connections
  between them, if they exists.

  You can also perform this operation using the ``&`` operator::

      model = (node1 >> node2) & (node1 >> node3))

  This is equivalent to::

      model = merge((node1 >> node2), (node1 >> node3))

  The inplace operator can also be used::

      model &= other_model

  Which is equivalent to::

      model.update_graph(other_model.nodes, other_model.edges)

  Parameters
  ----------
  node: Model or Node
      First node or model to merge. The `inplace` parameter takes this
      instance as reference.
  *other_nodes : Model or Node
      All models to merge.
  inplace: bool, default to False
      If `True`, then will update Model `model` inplace. If `model` is not
      a Model instance, this parameter will causes the function to raise
      a `ValueError`.
  name: str, optional
      Name of the resulting Model.

  Returns
  -------
  Model
      A new :py:class:`~.Model` instance.

  Raises
  ------
  ValueError
      If `inplace` is `True` but `model` is not a Model instance, then the
      operation is impossible. Inplace merging can only take place on a
      Model instance.
  """
  if not isinstance(node, Node):
    raise TypeError(f"Impossible to merge nodes: object {type(node)} is not a {Node.__name__} instance.")
  all_nodes = set()
  all_edges = set()
  for m in other_nodes:
    if isinstance(m, FrozenNetwork):
      raise TypeError(f'{FrozenNetwork.__name__} cannot merge with other nodes.')
    # fuse models nodes and edges (right side argument)
    if isinstance(m, Network):
      all_nodes |= set(m.lnodes)
      all_edges |= set(m.ff_edges)
    elif isinstance(m, Node):
      all_nodes.add(m)

  if inplace:
    if not isinstance(node, Network) or isinstance(node, FrozenNetwork):
      raise ValueError(f"Impossible to merge nodes inplace: "
                       f"{node} is not a {Network.__name__} instance.")
    return node.update_graph(tuple(all_nodes), tuple(all_edges))

  else:
    if isinstance(node, FrozenNetwork):
      raise TypeError(f'{FrozenNetwork.__name__} cannot merge with other nodes.')
    # add left side model nodes
    if isinstance(node, Network):
      all_nodes |= set(node.lnodes)
      all_edges |= set(node.ff_edges)
    elif isinstance(node, Node):
      all_nodes.add(node)
    return Network(nodes=tuple(all_nodes), ff_edges=tuple(all_edges), name=name)


def select(node: Node, index: slice, name: str = None, ):
  if isinstance(node, Network) and len(node.exit_nodes) != 1:
    raise ValueError(f'Cannot select subsets of states when Network instance '
                     f'"{node}" has multiple/zero output nodes.')
  return ff_connect(node, Select(index=index), name=name)
