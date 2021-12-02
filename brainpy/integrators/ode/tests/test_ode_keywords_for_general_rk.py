# -*- coding: utf-8 -*-

import unittest

import pytest

from brainpy import errors
from brainpy.integrators.ode import odeint


class TestExplicitRKKeywords(unittest.TestCase):
  def test_euler(self):
    print('Test Euler method:')
    print()
    odeint(method='euler', show_code=True, f=lambda v, t, p: t)

    with pytest.raises(errors.CodeError):
      odeint(method='euler', show_code=True, f=lambda f, t, dt: t)

    with pytest.raises(errors.CodeError):
      odeint(method='euler', show_code=True, f=lambda v, t, dt: t)

    with pytest.raises(errors.CodeError):
      odeint(method='euler', show_code=True, f=lambda v, t, v_new: t)

    with pytest.raises(errors.CodeError):
      odeint(method='euler', show_code=True, f=lambda v, t, dv_k1: t)

    print('-' * 40)

  def test_order2_rk(self):
    for method in ['heun2', 'midpoint', 'ralston2']:
      print(f'Test {method} method:')
      print()
      odeint(method=method, show_code=True, f=lambda v, t, p: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda f, t, dt: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dt: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, v_new: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dv_k1: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dv_k2: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k2_v_arg: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k2_t_arg: t)

      print('-' * 40)

  def test_rk2(self):
    method = 'rk2'

    print(f'Test {method} method:')
    print()
    odeint(method=method, show_code=True, f=lambda v, t, p: t)

    with pytest.raises(errors.CodeError):
      odeint(method=method, show_code=True, f=lambda f, t, dt: t)

    with pytest.raises(errors.CodeError):
      odeint(method=method, show_code=True, f=lambda v, t, dt: t)

    with pytest.raises(errors.CodeError):
      odeint(method=method, show_code=True, f=lambda v, t, v_new: t)

    with pytest.raises(errors.CodeError):
      odeint(method=method, show_code=True, f=lambda v, t, dv_k1: t)

    with pytest.raises(errors.CodeError):
      odeint(method=method, show_code=True, f=lambda v, t, dv_k2: t)

    with pytest.raises(errors.CodeError):
      odeint(method=method, show_code=True, f=lambda v, t, k2_v_arg: t)

    with pytest.raises(errors.CodeError):
      odeint(method=method, show_code=True, f=lambda v, t, k2_t_arg: t)

    print('-' * 40)

  def test_order3_rk(self):
    for method in ['rk3', 'heun3', 'ralston3', 'ssprk3']:
      print(f'Test {method} method:')
      print()
      odeint(method=method, show_code=True, f=lambda v, t, p: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda f, t, dt: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dt: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, v_new: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dv_k1: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dv_k2: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dv_k3: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k2_v_arg: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k2_t_arg: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k3_v_arg: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k3_t_arg: t)

      print('-' * 40)

  def test_order4_rk(self):
    for method in ['rk4', 'ralston4', 'rk4_38rule']:
      print(f'Test {method} method:')
      print()
      odeint(method=method, show_code=True, f=lambda v, t, p: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda f, t, dt: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dt: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, v_new: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dv_k1: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dv_k2: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dv_k3: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, dv_k4: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k2_v_arg: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k2_t_arg: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k3_v_arg: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k3_t_arg: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k4_v_arg: t)

      with pytest.raises(errors.CodeError):
        odeint(method=method, show_code=True, f=lambda v, t, k4_t_arg: t)

      print('-' * 40)
