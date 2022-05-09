# -*- coding: utf-8 -*-


__all__ = [
  'is_checking',
  'turn_on',
  'turn_off',
]

_check = True


def is_checking():
  """Whether the checking is turn on."""
  return _check


def turn_on():
  """Turn on the checking."""
  global _check
  _check = True


def turn_off():
  """Turn off the checking."""
  global _check
  _check = False
