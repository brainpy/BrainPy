# -*- coding: utf-8 -*-


from brainpy.errors import ConnectorError
from brainpy.tools import size2num

__all__ = [
  'get_pre_num', 'get_post_num',
  'get_pre_size', 'get_post_size',
]


def get_pre_size(obj, pre_size=None):
  if pre_size is None:
    if obj.pre_size is None:
      raise ConnectorError('Please provide "pre_size" and "post_size"')
    else:
      return obj.pre_size
  else:
    return (pre_size, ) if isinstance(pre_size, int) else pre_size


def get_pre_num(obj, pre_size=None):
  return size2num(get_pre_size(obj, pre_size))


def get_post_size(obj, post_size=None):
  if post_size is None:
    if obj.post_size is None:
      raise ConnectorError('Please provide "pre_size" and "post_size"')
    else:
      return obj.post_size
  else:
    return (post_size,) if isinstance(post_size, int) else post_size


def get_post_num(obj, post_size=None):
  return size2num(get_post_size(obj, post_size))
