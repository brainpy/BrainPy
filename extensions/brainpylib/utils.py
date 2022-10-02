# -*- coding: utf-8 -*-


__all__ = [
  'GPUOperatorNotFound',
]


class GPUOperatorNotFound(Exception):
  def __init__(self, name):
    super(GPUOperatorNotFound, self).__init__(f'''
GPU operator for "{name}" does not found. 

Please compile brainpylib GPU operators with the guidance in the following link:

https://brainpy.readthedocs.io/en/latest/tutorial_advanced/compile_brainpylib.html
    ''')

