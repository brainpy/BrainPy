# -*- coding: utf-8 -*-

from collections import deque, defaultdict

__all__ = [
  'find_parents_and_children',
  'find_entries_and_exits',
  'topological_sort',
]


def find_parents_and_children(edges):
  parents = defaultdict(list)  # find parents according to the child
  children = defaultdict(list)  # find children according to the parent
  for edge in edges:
    parent, child = edge
    parents[child].append(parent)
    children[parent].append(child)
  return parents, children


def find_entries_and_exits(nodes, edges):
  """Find input nodes and output nodes."""
  nodes = set(nodes)
  senders = set([n for n, _ in edges])
  receivers = set([n for _, n in edges])
  lonely = nodes - senders - receivers
  entry_points = senders - receivers | lonely
  end_points = receivers - senders | lonely
  return list(entry_points), list(end_points)


def topological_sort(nodes, edges, inputs=None):
  if inputs is None:
    inputs, _ = find_entries_and_exits(nodes, edges)
  parents, children = find_parents_and_children(edges)
  # using Kahn's algorithm
  ordered_nodes = []
  edges = set(edges)
  inputs = deque(inputs)
  while len(inputs) > 0:
    n = inputs.pop()
    ordered_nodes.append(n)
    for m in children.get(n, ()):
      edges.remove((n, m))
      parents[m].remove(n)
      if parents.get(m) is None or len(parents[m]) < 1:
        inputs.append(m)
  if len(edges) > 0:
    raise RuntimeError("Model has a cycle: impossible "
                       "to automatically determine operation "
                       "order in the model.")
  else:
    return ordered_nodes

