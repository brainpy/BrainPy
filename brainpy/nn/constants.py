# -*- coding: utf-8 -*-

from typing import Callable

__all__ = [
  'PASS_ONLY_ONE',
  'PASS_SEQUENCE',
  'PASS_NAME_DICT',
  'PASS_NODE_DICT',
  'DATA_PASS_TYPES',
  'DATA_PASS_FUNC',
  'register_data_pass_type',
]

"""Pass Type. Pass the only one data into the node. 
If there are multiple data, an error will be raised. 
"""
PASS_ONLY_ONE = 'PASS_ONLY_ONE'

"""Pass Type. Pass a list/tuple of data into the node."""
PASS_SEQUENCE = 'PASS_SEQUENCE'

"""Pass Type. Pass a dict with <node name, data> into the node."""
PASS_NAME_DICT = 'PASS_NAME_DICT'

"""Pass Type. Pass a dict with <node type, data> into the node."""
PASS_TYPE_DICT = 'PASS_TYPE_DICT'

"""Pass Type. Pass a dict with <node, data> into the node."""
PASS_NODE_DICT = 'PASS_NODE_DICT'

"""All supported data pass types."""
DATA_PASS_TYPES = [
  PASS_ONLY_ONE,
  PASS_SEQUENCE,
  PASS_NAME_DICT,
  PASS_TYPE_DICT,
  PASS_NODE_DICT,
]


def _pass_only_one(data):
  if data is None:
    return None
  if len(data) > 1:
    raise ValueError(f'"PASS_ONLY_ONE" type only support one '
                     f'feedforward/feedback input. But we got {len(data)}.')
  return tuple(data.values())[0]


def _pass_sequence(data):
  if data is None:
    return None
  else:
    return tuple(data.values())


def _pass_name_dict(data):
  if data is None:
    return data
  else:
    from brainpy.nn.base import Node
    _res = dict()
    for node, val in data.items():
      if isinstance(node, str):
        _res[node] = val
      elif isinstance(node, Node):
        _res[node.name] = val
      else:
        raise ValueError(f'Unknown type {type(node)}: node')
    return _res


def _pass_type_dict(data):
  if data is None:
    return data
  else:
    from brainpy.nn.base import Node
    _res = dict()
    for node, val in data.items():
      if isinstance(node, str):
        _res[str] = val
      elif isinstance(node, Node):
        _res[type(node.name)] = val
      else:
        raise ValueError(f'Unknown type {type(node)}: node')
    return _res


"""The conversion between the data pass type and 
the corresponding conversion function."""
DATA_PASS_FUNC = {
  PASS_SEQUENCE: _pass_sequence,
  PASS_NAME_DICT: _pass_name_dict,
  PASS_TYPE_DICT: _pass_type_dict,
  PASS_NODE_DICT: lambda a: a,
  PASS_ONLY_ONE: _pass_only_one,
}


def register_data_pass_type(name: str,
                            func: Callable):
  """Register a new data pass type.

  Parameters
  ----------
  name: str
    The data pass type name.
  func: callable
    The conversion function of the data pass type.
  """
  if name in DATA_PASS_TYPES:
    raise ValueError(f'Data pass type "{name}" has been registered.')
  DATA_PASS_FUNC[name] = func
