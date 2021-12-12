# -*- coding: utf-8 -*-

import sys

__all__ = [
  'output',
]


def output(msg, file=None):
  if file is None:
    file = sys.stderr
  print(msg, file=file)
