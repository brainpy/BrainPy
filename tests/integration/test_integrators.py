# -*- coding: utf-8 -*-

import numpy as np
from brainpy.integration import integrate
from brainpy.integration import integrator


def test_exponential_euler_by_linear_system():
    @integrate(method='exponential')
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return (dmdt,), alpha, beta

    assert isinstance(int_m, integrator.ExponentialEuler)
    print(int_m._update_code)
    assert int_m._update_code == '''
_alpha = 0.1 * (_V + 40) / (1 - np.exp(-(_V + 40) / 10))
_beta = 4.0 * np.exp(-(_V + 65) / 18)
_dfm_dt = -_alpha*_m + 1.0*_alpha - _beta*_m
_m_linear = -_alpha - _beta
_m_linear_exp = exp(-0.1*_alpha - 0.1*_beta)
_m_df_part = _dfm_dt*(_m_linear_exp - 1)/_m_linear
_m = _m_df_part + _m
_res = _m, _alpha, _beta'''.strip()


def test_exponential_euler_by_nonlinear_system():
    @integrate(method='exponential')
    def int_v(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext

    assert isinstance(int_v, integrator.ExponentialEuler)
    print(int_v._update_code)
    assert int_v._update_code == '''
_dfv_dt = _Iext - 0.333333333333333*_v**3 + _v - _w
_v_linear = 1
_v_linear_exp = 1.10517091807565
_v_df_part = _dfv_dt*(_v_linear_exp - 1)/_v_linear
_v = _v_df_part + _v
_res = _v'''.strip()


def test_exponential_euler_by_linear_system_with_noise():
    @integrate(method='exponential')
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return (dmdt, alpha), alpha, beta

    assert isinstance(int_m, integrator.ExponentialEuler)
    print(int_m._update_code)
    assert int_m._update_code == '''
_alpha = 0.1 * (_V + 40) / (1 - np.exp(-(_V + 40) / 10))
_beta = 4.0 * np.exp(-(_V + 65) / 18)
_dfm_dt = -_alpha*_m + 1.0*_alpha - _beta*_m
_m_linear = -_alpha - _beta
_m_linear_exp = exp(-0.1*_alpha - 0.1*_beta)
_m_df_part = _dfm_dt*(_m_linear_exp - 1)/_m_linear
_m_dW = _normal_like_(_m)
_alpha = 0.1 * (_V + 40) / (1 - np.exp(-(_V + 40) / 10))
_m_dg_part = _alpha * _m_dW
_m = _m_df_part + _m_dg_part*_m_linear_exp + _m
_res = _m, _alpha, _beta'''.strip()


def test_euler():
    @integrate(method='euler')
    def int_m(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext

    assert isinstance(int_m, integrator.Euler)
    print(int_m._update_code)


def test_heun():
    @integrate(method='heun')
    def int_m(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext

    assert isinstance(int_m, integrator.Heun)
    print(int_m._update_code)


def test_midpoint():
    @integrate(method='midpoint')
    def int_m(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext

    assert isinstance(int_m, integrator.MidPoint)
    print(int_m._update_code)

def test_rk2():
    @integrate(method='rk2')
    def int_m(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext

    assert isinstance(int_m, integrator.RK2)
    print(int_m._update_code)

def test_rk3():
    @integrate(method='rk3')
    def int_m(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext

    assert isinstance(int_m, integrator.RK3)
    print(int_m._update_code)



def test_rk4():
    @integrate(method='rk4')
    def int_m(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext

    assert isinstance(int_m, integrator.RK4)
    print(int_m._update_code)


def test_rk4_multi_system():
    @integrate(method='exponential')
    def int_func(array, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        m = alpha * (1 - array[0]) - beta * array[0]

        alpha2 = 0.07 * np.exp(-(V + 65) / 20.)
        beta2 = 1 / (1 + np.exp(-(V + 35) / 10))
        h = alpha2 * (1 - array[1]) - beta2 * array[1]
        return np.array([m, h])

    assert isinstance(int_func, integrator.RK4)
    print(int_func._update_code)





def test_rk4_alternative():
    @integrate(method='rk4_alternative')
    def int_m(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext

    assert isinstance(int_m, integrator.RK4Alternative)
    print(int_m._update_code)


def test_milstein_ito_without_noise():
    @integrate(method='milstein_ito')
    def int_m(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext

    assert isinstance(int_m, integrator.MilsteinIto)
    print(int_m._update_code)


def test_milstein_ito_with_noise():
    @integrate(method='milstein_ito')
    def int_m(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext, -v * v

    assert isinstance(int_m, integrator.MilsteinIto)
    print(int_m._update_code)


def test_milstein_stra_with_noise():
    @integrate(method='milstein_stra')
    def int_m(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext, -v * v

    assert isinstance(int_m, integrator.MilsteinStra)
    print(int_m._update_code)




if __name__ == '__main__':
    # test_exponential_euler_by_linear_system()
    # test_exponential_euler_by_linear_system_with_noise()
    # test_euler()
    # test_heun()
    # test_midpoint()
    # test_rk2()
    # test_rk3()
    # test_rk4()
    test_rk4_multi_system()
    # test_rk4_alternative()
    # test_milstein_ito_without_noise()
    # test_milstein_ito_with_noise()
    # test_milstein_stra_with_noise()
