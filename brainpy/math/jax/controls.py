# -*- coding: utf-8 -*-

from jax import lax

__all__ = [
  'scan',
  'fori_loop',
  'while_loop',
  'cond',
]


def scan(f, init, xs, length=None, reverse=False, unroll=1):
  return lax.scan(f=f, init=init, xs=xs, length=length, reverse=reverse, unroll=unroll)


def fori_loop(lower, upper, body_fun, init_val):
  return lax.fori_loop(lower=lower, upper=upper, body_fun=body_fun, init_val=init_val)


def while_loop(cond_fun, body_fun, init_val):
  return lax.while_loop(cond_fun, body_fun, init_val)


def cond(pred, true_fun, false_fun, operand):
  return lax.cond(pred, true_fun, false_fun, operand)
