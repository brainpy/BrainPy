# -*- coding: utf-8 -*-

import unittest

import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy.integrators.ode.exponential import ExponentialEuler


class TestExpnentialEuler(unittest.TestCase):
  def test_hh_model(self):
    def drivative(V, m, h, n, t, Iext, gNa, ENa, gK, EK, gL, EL, C):
      alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
      beta = 4.0 * bm.exp(-(V + 65) / 18)
      dmdt = alpha * (1 - m) - beta * m

      alpha = 0.07 * bm.exp(-(V + 65) / 20.)
      beta = 1 / (1 + bm.exp(-(V + 35) / 10))
      dhdt = alpha * (1 - h) - beta * h

      alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
      beta = 0.125 * bm.exp(-(V + 65) / 80)
      dndt = alpha * (1 - n) - beta * n

      I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
      I_K = (gK * n ** 4.0) * (V - EK)
      I_leak = gL * (V - EL)
      dVdt = (- I_Na - I_K - I_leak + Iext) / C

      return dVdt, dmdt, dhdt, dndt

    ExponentialEuler(f=drivative, show_code=True, dt=0.01, var_type='SCALAR')

  def test_return_expr(self):
    def derivative(s, t, tau):
      return -s / tau

    with pytest.raises(bp.errors.DiffEqError):
      ExponentialEuler(f=derivative, show_code=True, dt=0.01, var_type='SCALAR', )

  def test_return_expr2(self):
    def derivative(s, v, t, tau):
      dv = -v + 1
      return -s / tau, dv

    with pytest.raises(bp.errors.DiffEqError):
      ExponentialEuler(f=derivative, show_code=True, dt=0.01, var_type='SCALAR', )

  def test_return_expr3(self):
    f = lambda s, t, tau: -s / tau
    with pytest.raises(bp.errors.AnalyzerError) as excinfo:
      ExponentialEuler(f=f, show_code=True, dt=0.01, var_type='SCALAR', )

  def test_nonlinear_eq1_vdp(self):
    def vdp_derivative(x, y, t, mu):
      dx = mu * (x - x ** 3 / 3 - y)
      dy = x / mu
      return dx, dy

    ExponentialEuler(f=vdp_derivative, show_code=True, dt=0.01)

  def test_nonlinear_eq2_reduced_trn(self):
    T = 36.
    phi_m = phi_h = phi_n = 3 ** ((T - 36) / 10)
    # parameters of IT
    E_T = 120.
    phi_p = 5 ** ((T - 24) / 10)
    phi_q = 3 ** ((T - 24) / 10)
    p_half, p_k = -52., 7.4
    q_half, q_k = -80., -5.
    g_Na = 100.
    E_Na = 50.
    g_K = 10.
    # parameters of V
    C, Vth, area = 1., 20., 1.43e-4
    V_factor = 1e-3 / area

    def reduced_trn_derivative(V, y, z, t, Isyn, b, rho_p, g_T, g_L, g_KL, E_L, E_KL, IT_th, NaK_th):
      # m channel
      t1 = 13. - V + NaK_th
      t1_exp = bm.exp(t1 / 4.)
      m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
      m_alpha_by_V_diff = (-0.32 * (t1_exp - 1.) + 0.08 * t1 * t1_exp) / (t1_exp - 1.) ** 2  # \alpha_m'(V)
      t2 = V - 40. - NaK_th
      t2_exp = bm.exp(t2 / 5.)
      m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
      m_beta_by_V_diff = (0.28 * (t2_exp - 1) - 0.056 * t2 * t2_exp) / (t2_exp - 1) ** 2  # \beta_m'(V)
      m_tau_by_V = 1. / phi_m / (m_alpha_by_V + m_beta_by_V)  # \tau_m(V)
      m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)
      m_inf_by_V_diff = (m_alpha_by_V_diff * m_beta_by_V - m_alpha_by_V * m_beta_by_V_diff) / \
                        (m_alpha_by_V + m_beta_by_V) ** 2  # \m_{\infty}'(V)

      # h channel
      h_alpha_by_V = 0.128 * bm.exp((17. - V + NaK_th) / 18.)  # \alpha_h(V)
      h_beta_by_V = 4. / (bm.exp((40. - V + NaK_th) / 5.) + 1.)  # \beta_h(V)
      h_inf_by_V = h_alpha_by_V / (h_alpha_by_V + h_beta_by_V)  # h_{\infty}(V)
      h_tau_by_V = 1. / phi_h / (h_alpha_by_V + h_beta_by_V)  # \tau_h(V)
      h_alpha_by_y = 0.128 * bm.exp((17. - y + NaK_th) / 18.)  # \alpha_h(y)
      t3 = bm.exp((40. - y + NaK_th) / 5.)
      h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
      h_beta_by_y_diff = 0.8 * t3 / (1 + t3) ** 2  # \beta_h'(y)
      h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # h_{\infty}(y)
      h_alpha_by_y_diff = - h_alpha_by_y / 18.  # \alpha_h'(y)
      h_inf_by_y_diff = (h_alpha_by_y_diff * h_beta_by_y - h_alpha_by_y * h_beta_by_y_diff) / \
                        (h_beta_by_y + h_alpha_by_y) ** 2  # h_{\infty}'(y)

      # n channel
      t4 = (15. - V + NaK_th)
      n_alpha_by_V = 0.032 * t4 / (bm.exp(t4 / 5.) - 1.)  # \alpha_n(V)
      n_beta_by_V = b * bm.exp((10. - V + NaK_th) / 40.)  # \beta_n(V)
      n_tau_by_V = 1. / (n_alpha_by_V + n_beta_by_V) / phi_n  # \tau_n(V)
      n_inf_by_V = n_alpha_by_V / (n_alpha_by_V + n_beta_by_V)  # n_{\infty}(V)
      t5 = (15. - y + NaK_th)
      t5_exp = bm.exp(t5 / 5.)
      n_alpha_by_y = 0.032 * t5 / (t5_exp - 1.)  # \alpha_n(y)
      t6 = bm.exp((10. - y + NaK_th) / 40.)
      n_beta_y = b * t6  # \beta_n(y)
      n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_y)  # n_{\infty}(y)
      n_alpha_by_y_diff = (0.0064 * t5 * t5_exp - 0.032 * (t5_exp - 1.)) / (t5_exp - 1.) ** 2  # \alpha_n'(y)
      n_beta_by_y_diff = -n_beta_y / 40  # \beta_n'(y)
      n_inf_by_y_diff = (n_alpha_by_y_diff * n_beta_y - n_alpha_by_y * n_beta_by_y_diff) / \
                        (n_alpha_by_y + n_beta_y) ** 2  # n_{\infty}'(y)

      # p channel
      p_inf_by_V = 1. / (1. + bm.exp((p_half - V + IT_th) / p_k))  # p_{\infty}(V)
      p_tau_by_V = (3 + 1. / (bm.exp((V + 27. - IT_th) / 10.) +
                              bm.exp(-(V + 102. - IT_th) / 15.))) / phi_p  # \tau_p(V)
      t7 = bm.exp((p_half - y + IT_th) / p_k)
      p_inf_by_y = 1. / (1. + t7)  # p_{\infty}(y)
      p_inf_by_y_diff = t7 / p_k / (1. + t7) ** 2  # p_{\infty}'(y)

      # p channel
      q_inf_by_V = 1. / (1. + bm.exp((q_half - V + IT_th) / q_k))  # q_{\infty}(V)
      t8 = bm.exp((q_half - z + IT_th) / q_k)
      q_inf_by_z = 1. / (1. + t8)  # q_{\infty}(z)
      q_inf_diff_z = t8 / q_k / (1. + t8) ** 2  # q_{\infty}'(z)
      q_tau_by_V = (85. + 1 / (bm.exp((V + 48. - IT_th) / 4.) +
                               bm.exp(-(V + 407. - IT_th) / 50.))) / phi_q  # \tau_q(V)

      # ----
      #  x
      # ----

      gNa = g_Na * m_inf_by_V ** 3 * h_inf_by_y  # gNa
      gK = g_K * n_inf_by_y ** 4  # gK
      gT = g_T * p_inf_by_y * p_inf_by_y * q_inf_by_z  # gT
      FV = gNa + gK + gT + g_L + g_KL  # dF/dV
      Fm = 3 * g_Na * h_inf_by_y * (V - E_Na) * m_inf_by_V * m_inf_by_V * m_inf_by_V_diff  # dF/dvm
      t9 = C / m_tau_by_V
      t10 = FV + Fm
      t11 = t9 + FV
      rho_V = (t11 - bm.sqrt(bm.maximum(t11 ** 2 - 4 * t9 * t10, 0.))) / 2 / t10  # rho_V
      INa = gNa * (V - E_Na)
      IK = gK * (V - E_KL)
      IT = gT * (V - E_T)
      IL = g_L * (V - E_L)
      IKL = g_KL * (V - E_KL)
      Iext = V_factor * Isyn
      dVdt = rho_V * (-INa - IK - IT - IL - IKL + Iext) / C

      # ----
      #  y
      # ----

      Fvh = g_Na * m_inf_by_V ** 3 * (V - E_Na) * h_inf_by_y_diff  # dF/dvh
      Fvn = 4 * g_K * (V - E_KL) * n_inf_by_y ** 3 * n_inf_by_y_diff  # dF/dvn
      f4 = Fvh + Fvn
      rho_h = (1 - rho_p) * Fvh / f4
      rho_n = (1 - rho_p) * Fvn / f4
      fh = (h_inf_by_V - h_inf_by_y) / h_tau_by_V / h_inf_by_y_diff
      fn = (n_inf_by_V - n_inf_by_y) / n_tau_by_V / n_inf_by_y_diff
      fp = (p_inf_by_V - p_inf_by_y) / p_tau_by_V / p_inf_by_y_diff
      dydt = rho_h * fh + rho_n * fn + rho_p * fp

      # ----
      #  z
      # ----

      dzdt = (q_inf_by_V - q_inf_by_z) / q_tau_by_V / q_inf_diff_z

      return dVdt, dydt, dzdt

    with pytest.raises(bp.errors.DiffEqError):
      ExponentialEuler(f=reduced_trn_derivative, show_code=True, dt=0.01, timeout=5)

  def test_nonlinear_eq3_adaptive_quadratic_if(self):
    def derivative(V, w, t, Iext, self):
      dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) - w + Iext) / self.tau
      dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
      return dVdt, dwdt

    ExponentialEuler(f=derivative, show_code=True, dt=0.01, timeout=5)

  def test_nonlinear_eq4_exponentil_if(self):
    def derivative(V, t, Iext, self):
      exp_v = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
      dvdt = (- (V - self.V_rest) + exp_v + self.R * Iext) / self.tau
      return dvdt

    ExponentialEuler(f=derivative, show_code=True, dt=0.01, timeout=5)

  def test_nonlinear_eq5_morris_lecar(self):
    def derivative(V, W, t, I_ext, self):
      M_inf = (1 / 2) * (1 + bm.tanh((V - self.V1) / self.V2))
      I_Ca = self.g_Ca * M_inf * (V - self.V_Ca)
      I_K = self.g_K * W * (V - self.V_K)
      I_Leak = self.g_leak * (V - self.V_leak)
      dVdt = (- I_Ca - I_K - I_Leak + I_ext) / self.C

      tau_W = 1 / (self.phi * bm.cosh((V - self.V3) / (2 * self.V4)))
      W_inf = (1 / 2) * (1 + bm.tanh((V - self.V3) / self.V4))
      dWdt = (W_inf - W) / tau_W
      return dVdt, dWdt

    ExponentialEuler(f=derivative, show_code=True, dt=0.01, timeout=5)

  def test1(self):
    def dev(x, t):
      dx = bm.power(x, 3)
      return dx

    ExponentialEuler(f=dev, show_code=True, dt=0.01, timeout=5)

