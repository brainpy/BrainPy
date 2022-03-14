# -*- coding: utf-8 -*-


"""
This module provides basic tool for graphs, including

- detect the senders and receivers in the network graph,
- find input and output nodes in a given graph,
- detect the cycle in the graph,
- detect the path between two nodes.

"""


from collections import deque, defaultdict

__all__ = [
  'find_senders_and_receivers',
  'find_entries_and_exits',
  'detect_cycle',
  'detect_path',
]


def find_senders_and_receivers(edges):
  """Find all senders and receivers in the given graph."""
  senders = dict()  # find parents according to the child
  receivers = dict()  # find children according to the parent
  for edge in edges:
    sender, receiver = edge
    if receiver not in senders:
      senders[receiver] = [sender]
    else:
      senders[receiver].append(sender)
    if sender not in receivers:
      receivers[sender] = [receiver]
    else:
      receivers[sender].append(receiver)
  return senders, receivers


def find_entries_and_exits(nodes, ff_edges, fb_edges=()):
  """Find input nodes and output nodes."""
  nodes = set(nodes)
  ff_senders = set([n for n, _ in ff_edges])
  ff_receivers = set([n for _, n in ff_edges])
  fb_senders = set([n for n, _ in fb_edges])
  fb_receivers = set([n for _, n in fb_edges])

  # # check lonely feedback nodes
  # fb_receivers_without_ff = fb_receivers - ff_receivers - ff_senders
  # if len(fb_receivers_without_ff) > 0:
  #   raise ValueError(f'Found feedback nodes do not define feedforward connections: \n\n'
  #                    f'{fb_receivers_without_ff}')

  # check lonely nodes
  lonely = nodes - ff_senders - ff_receivers - fb_senders - fb_receivers
  # if len(lonely):
  #   _str_nodes = '\n'.join([str(node) for node in lonely])
  #   raise ValueError(f"Found lonely nodes \n\n{_str_nodes} \n\n"
  #                    f"which do not connect with any other.")

  # get input and output nodes
  entry_points = (ff_senders | fb_senders) - ff_receivers - lonely
  end_points = ff_receivers - ff_senders - lonely
  return list(entry_points), list(end_points)


def topological_sort(nodes, ff_edges, inputs=None):
  if inputs is None:
    inputs, _ = find_entries_and_exits(nodes, ff_edges)
  parents, children = find_senders_and_receivers(ff_edges)
  # using Kahn's algorithm
  ordered_nodes = []
  ff_edges = set(ff_edges)
  inputs = deque(inputs)
  while len(inputs) > 0:
    n = inputs.pop()
    ordered_nodes.append(n)
    for m in children.get(n, ()):
      ff_edges.remove((n, m))
      parents[m].remove(n)
      if parents.get(m) is None or len(parents[m]) < 1:
        inputs.append(m)
  if len(ff_edges) > 0:
    raise RuntimeError("Model has a cycle: impossible "
                       "to automatically determine operation "
                       "order in the model.")
  else:
    return ordered_nodes


def _detect_cycle(v, visited, stacks, graph):
  # visited数组元素为true，标记该元素被isCyclicUtil递归调用链处理中，或处理过
  # recStack数组元素为true，表示该元素还在递归函数isCyclicUtil的函数栈中
  visited[v] = True
  stacks[v] = True
  # 深度遍历所有节点。
  for neighbour in graph[v]:
    if not visited[neighbour]:  # 如果该节点没有被处理过，那么继续调用递归
      if _detect_cycle(neighbour, visited, stacks, graph):  # 如果邻接点neighbour的递归发现了环
        return True  # 那么返回真
    elif stacks[neighbour]:  # 如果neighbour被处理中（这里强调了不是处理过），且还在递归栈中，说明发现了环
      return True
  stacks[v] = False  # 函数开始时，V节点进栈。所以函数结束时，V节点出栈。
  return False  # v的所有邻接点的递归都没有发现环，则返回假


def detect_cycle(nodes, edges):
  """Detect whether a cycle exists in the defined graph.
  """
  node2id = {node: i for i, node in enumerate(nodes)}
  graph = defaultdict(list)
  for s, r in edges:
    graph[node2id[s]].append(node2id[r])
  num = len(nodes)

  visited = [False] * num
  stacks = [False] * num
  for i in range(num):  # 分别以每个节点作为起点，然后开始深度遍历
    if not visited[i]:  # 这里为真，说明之前的深度遍历已经遍历过该节点了，且那次遍历没有发现环
      if _detect_cycle(i, visited, stacks, graph):  # 如果发现环，直接返回
        return True
  return False  # 如果分别以每个节点作为起点的深度遍历都没有发现环，那肯定是整个图没有环


def _has_path_by_dfs(from_node, to_node, graph):
  # queue本质上是堆栈，用来存放需要进行遍历的数据
  # order里面存放的是具体的访问路径
  queue, order = [], []
  # 首先将初始遍历的节点放到queue中，表示将要从这个点开始遍历
  queue.append(from_node)
  while len(queue):
    # 从queue中pop出点v，然后从v点开始遍历了，所以可以将这个点pop出，然后将其放入order中
    # 这里才是最有用的地方，pop（）表示弹出栈顶，由于下面的for循环不断的访问子节点，并将子节点压入堆栈，
    # 也就保证了每次的栈顶弹出的顺序是下面的节点
    v = queue.pop()
    order.append(v)
    # 这里开始遍历v的子节点
    for w in graph[v]:
      # w既不属于queue也不属于order，意味着这个点没被访问过，所以讲起放到queue中，然后后续进行访问
      if w not in order and w not in queue:
        if w == to_node:
          return True
        else:
          queue.append(w)
  return False


def _has_path_by_bfs(from_node, to_node, graph):
  # queue本质上是堆栈，用来存放需要进行遍历的数据
  # order里面存放的是具体的访问路径
  queue, order = [], []
  # 首先将初始遍历的节点放到queue中，表示将要从这个点开始遍历
  # 由于是广度优先，也就是先访问初始节点的所有的子节点，所以可以
  queue.append(from_node)
  order.append(from_node)
  while len(queue):
    # queue.pop(0)意味着是队列的方式出元素，就是先进先出，而下面的for循环将节点v的所有子节点
    # 放到queue中，所以queue.pop(0)就实现了每次访问都是先将元素的子节点访问完毕，而不是优先叶子节点
    v = queue.pop(0)
    for w in graph[v]:
      if w not in order:
        if w == to_node:
          return True
        else:
          # 这里可以直接order.append(w) 因为广度优先就是先访问节点的所有下级子节点，所以可以
          # 将self.sequense[v]的值直接全部先给到order
          order.append(w)
          queue.append(w)
  return False


def detect_path(from_node, to_node, edges, method='dfs'):
  """Detect whether there is a path exist in the defined graph
  from ``from_node`` to ``to_node``. """
  graph = defaultdict(list)
  for s, r in edges:
    graph[s].append(r)
  if method == 'dfs':
    return _has_path_by_dfs(from_node, to_node, graph)
  elif method == 'bfs':
    return _has_path_by_bfs(from_node, to_node, graph)
  else:
    raise ValueError(f'Unknown method {method}')
