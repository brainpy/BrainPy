# -*- coding: utf-8 -*-

from brainpy.math.numpy import ops

import numba.misc.help.inspector as inspector


def test():
  supported = []
  unsupported = []
  for k in dir(ops):
    if not k.startswith('__'):
      v = getattr(ops, k)
      if isinstance(v, type(test)):
        r = inspector.inspect_function(v)
        if r['numba_type'] is None:
          print(v)
          unsupported.append(k)
        else:
          supported.append(k)
  print('Supported:')
  print(supported)
  print('unsupported:')
  print(unsupported)