# -*- coding: utf-8 -*-

import pytest

from brainpy import errors
from brainpy.integrators import ode


def test_bs():
  method = 'bs'
  for adaptive in [True, False]:

    print(f'Test {"adaptive" if adaptive else ""} {method} method:')
    print()
    ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_t_arg: t)

    if adaptive:
      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

    print('-' * 40)


def test_rkf45():
  method = 'rkf45'
  for adaptive in [True, False]:

    print(f'Test {"adaptive" if adaptive else ""} {method} method:')
    print()
    ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_t_arg: t)

    if adaptive:
      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

    print('-' * 40)


def test_heun_euler():
  method = 'heun_euler'
  for adaptive in [True, False]:

    print(f'Test {"adaptive" if adaptive else ""} {method} method:')
    print()
    ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

    if adaptive:
      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

    print('-' * 40)


def test_rkf12():
  method = 'rkf12'
  for adaptive in [True, False]:

    print(f'Test {"adaptive" if adaptive else ""} {method} method:')
    print()
    ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_t_arg: t)

    if adaptive:
      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

    print('-' * 40)


def test_ck():
  method = 'ck'
  for adaptive in [True, False]:

    print(f'Test {"adaptive" if adaptive else ""} {method} method:')
    print()
    ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k5_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k5_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k6_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k6_t_arg: t)

    if adaptive:
      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

    print('-' * 40)


def test_rkdp():
  method = 'rkdp'
  for adaptive in [True, False]:

    print(f'Test {"adaptive" if adaptive else ""} {method} method:')
    print()
    ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k5_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k5_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k6_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k6_t_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k7_v_arg: t)

    with pytest.raises(errors.CodeError):
      ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k7_t_arg: t)

    if adaptive:
      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

      with pytest.raises(errors.CodeError):
        ode.odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

    print('-' * 40)
