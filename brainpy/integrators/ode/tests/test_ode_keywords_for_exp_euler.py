# -*- coding: utf-8 -*-

import numpy as np
import pytest

from brainpy import errors
from brainpy.integrators.ode import odeint


def test_exp_euler():
  method = 'exponential_euler'

  print(f'Test {method} method:')
  print()

  def func(m, t, V):
    alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta = 4.0 * np.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m
    return dmdt

  odeint(method=method, show_code=True, f=func)

  with pytest.raises(errors.CodeError):
    def func(f, t, V):
      alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
      beta = 4.0 * np.exp(-(V + 65) / 18)
      dmdt = alpha * (1 - f) - beta * f
      return dmdt

    odeint(method=method, show_code=True, f=func)

  with pytest.raises(errors.CodeError):
    def func(m, t, dt):
      alpha = 0.1 * (dt + 40) / (1 - np.exp(-(dt + 40) / 10))
      beta = 4.0 * np.exp(-(dt + 65) / 18)
      dmdt = alpha * (1 - m) - beta * m
      return dmdt

    odeint(method=method, show_code=True, f=func)

  with pytest.raises(errors.CodeError):
    def func(m, t, m_new):
      alpha = 0.1 * (m_new + 40) / (1 - np.exp(-(m_new + 40) / 10))
      beta = 4.0 * np.exp(-(m_new + 65) / 18)
      dmdt = alpha * (1 - m) - beta * m
      return dmdt

    odeint(method=method, show_code=True, f=func)

  with pytest.raises(errors.CodeError):
    def func(m, t, exp):
      alpha = 0.1 * (exp + 40) / (1 - np.exp(-(exp + 40) / 10))
      beta = 4.0 * np.exp(-(exp + 65) / 18)
      dmdt = alpha * (1 - m) - beta * m
      return dmdt

    odeint(method=method, show_code=True, f=func)

  print('-' * 40)

