# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pytest

from brainpy import errors
from brainpy import odeint


class TestExponentialEuler(unittest.TestCase):
  def test1(self):
    def func(m, t, V):
      alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
      beta = 4.0 * np.exp(-(V + 65) / 18)
      dmdt = alpha * (1 - m) - beta * m
      return dmdt

    odeint(method='exponential_euler', show_code=True, f=func)

  def test3(self):
    with pytest.raises(errors.CodeError):
      def func(m, t, dt):
        alpha = 0.1 * (dt + 40) / (1 - np.exp(-(dt + 40) / 10))
        beta = 4.0 * np.exp(-(dt + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return dmdt

      odeint(method='exponential_euler', show_code=True, f=func)
