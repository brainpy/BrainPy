# -*- coding: utf-8 -*-


from brainpy import errors


def set(backend=None, dt=None):
  raise errors.NoLongerSupportError('"brainpy.backend" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.use_backend(xx) \n'
                                    '>>> # and \n'
                                    '>>> brainpy.math.set_dt(xx)\n')

