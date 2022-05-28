# -*- coding: utf-8 -*-


__all__ = [
  # data types
  'DataType',

  # pass rules
  'SingleData',
  'MultipleData',
]


class DataType(object):
  """Base class for data type."""

  def filter(self, data):
    raise NotImplementedError

  def __repr__(self):
    return self.__class__.__name__


class SingleData(DataType):
  """Pass the only one data into the node.
  If there are multiple data, an error will be raised. """

  def filter(self, data):
    if data is None:
      return None
    if len(data) > 1:
      raise ValueError(f'{self.__class__.__name__} only support one '
                       f'feedforward/feedback input. But we got {len(data)}.')
    return tuple(data.values())[0]

  def __repr__(self):
    return self.__class__.__name__


class MultipleData(DataType):
  """Pass a list/tuple of data into the node."""

  def __init__(self, return_type: str = 'sequence'):
    if return_type not in ['sequence', 'name_dict', 'type_dict', 'node_dict']:
      raise ValueError(f"Only support return type of 'sequence', 'name_dict', "
                       f"'type_dict' and 'node_dict'. But we got {return_type}")
    self.return_type = return_type

    from brainpy.nn.base import Node

    if return_type == 'sequence':
      f = lambda data: tuple(data.values())

    elif return_type == 'name_dict':
      # Pass a dict with <node name, data> into the node.

      def f(data):
        _res = dict()
        for node, val in data.items():
          if isinstance(node, str):
            _res[node] = val
          elif isinstance(node, Node):
            _res[node.name] = val
          else:
            raise ValueError(f'Unknown type {type(node)}: node')
        return _res

    elif return_type == 'type_dict':
      # Pass a dict with <node type, data> into the node.

      def f(data):
        _res = dict()
        for node, val in data.items():
          if isinstance(node, str):
            _res[str] = val
          elif isinstance(node, Node):
            _res[type(node.name)] = val
          else:
            raise ValueError(f'Unknown type {type(node)}: node')
        return _res

    elif return_type == 'node_dict':
      # Pass a dict with <node, data> into the node.
      f = lambda data: data

    else:
      raise ValueError
    self.return_func = f

  def __repr__(self):
    return f'{self.__class__.__name__}(return_type={self.return_type})'

  def filter(self, data):
    if data is None:
      return None
    else:
      return self.return_func(data)
