# -*- coding: utf-8 -*-

from brainpy.tools.others import init_progress_bar

from jax import lax
import numpy as np


def try_progress_bar():
  xs = np.random.rand(100002)
  bar = init_progress_bar(duration=len(xs), dt=1, )

  def cumsum(data, el):
    """
    - `res`: The result from the previous loop.
    - `el`: The current array element.
    """
    acc, i = data
    bar(i)
    acc = acc + el
    return (acc, i + 1), acc  # ("carryover", "accumulated")

  result_init, idx = 0, 0
  final, result = lax.scan(cumsum, (result_init, idx), xs)


if __name__ == '__main__':
  try_progress_bar()
