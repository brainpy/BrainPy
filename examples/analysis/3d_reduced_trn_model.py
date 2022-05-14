# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

bp.math.enable_x64()
bp.math.set_platform('cpu')


class ReducedTRNModel(bp.dyn.NeuGroup):
  def __init__(self, size, name=None, T=36., method='rk4'):
    super(ReducedTRNModel, self).__init__(size=size, name=name)

    self.IT_th = -3.
    self.b = 0.5
    self.g_T = 2.0
    self.g_L = 0.02
    self.E_L = -70.
    self.g_KL = 0.005
    self.E_KL = -95.
    self.NaK_th = -55.

    # temperature
    self.T = T

    # parameters of INa, IK
    self.g_Na = 100.
    self.E_Na = 50.
    self.g_K = 10.
    self.phi_m = self.phi_h = self.phi_n = 3 ** ((self.T - 36) / 10)

    # parameters of IT
    self.E_T = 120.
    self.phi_p = 5 ** ((self.T - 24) / 10)
    self.phi_q = 3 ** ((self.T - 24) / 10)
    self.p_half, self.p_k = -52., 7.4
    self.q_half, self.q_k = -80., -5.

    # parameters of V
    self.C, self.Vth, self.area = 1., 20., 1.43e-4
    self.V_factor = 1e-3 / self.area

    # parameters
    self.b = 0.14
    self.rho_p = 0.

    # variables
    self.V = bm.Variable(bm.zeros(self.num))
    self.y = bm.Variable(bm.zeros(self.num))
    self.z = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.input = bm.Variable(bm.zeros(self.num))

    # functions
    self.int_V = bp.odeint(self.fV, method=method)
    self.int_y = bp.odeint(self.fy, method=method)
    self.int_z = bp.odeint(self.fz, method=method)
    if not isinstance(self.int_V, bp.ode.ExponentialEuler):
      self.integral = bp.odeint(self.derivative, method=method)

  def fV(self, V, t, y, z, Isyn):
    # m channel
    t1 = 13. - V + self.NaK_th
    t1_exp = bm.exp(t1 / 4.)
    m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
    m_alpha_by_V_diff = (-0.32 * (t1_exp - 1.) + 0.08 * t1 * t1_exp) / (t1_exp - 1.) ** 2  # \alpha_m'(V)
    t2 = V - 40. - self.NaK_th
    t2_exp = bm.exp(t2 / 5.)
    m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
    m_beta_by_V_diff = (0.28 * (t2_exp - 1) - 0.056 * t2 * t2_exp) / (t2_exp - 1) ** 2  # \beta_m'(V)
    m_tau_by_V = 1. / self.phi_m / (m_alpha_by_V + m_beta_by_V)  # \tau_m(V)
    m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)
    m_inf_by_V_diff = (m_alpha_by_V_diff * m_beta_by_V - m_alpha_by_V * m_beta_by_V_diff) / \
                      (m_alpha_by_V + m_beta_by_V) ** 2  # \m_{\infty}'(V)

    # h channel
    h_alpha_by_y = 0.128 * bm.exp((17. - y + self.NaK_th) / 18.)  # \alpha_h(y)
    t3 = bm.exp((40. - y + self.NaK_th) / 5.)
    h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
    h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # h_{\infty}(y)

    # n channel
    t5 = (15. - y + self.NaK_th)
    t5_exp = bm.exp(t5 / 5.)
    n_alpha_by_y = 0.032 * t5 / (t5_exp - 1.)  # \alpha_n(y)
    t6 = bm.exp((10. - y + self.NaK_th) / 40.)
    n_beta_y = self.b * t6  # \beta_n(y)
    n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_y)  # n_{\infty}(y)

    # p channel
    t7 = bm.exp((self.p_half - y + self.IT_th) / self.p_k)
    p_inf_by_y = 1. / (1. + t7)  # p_{\infty}(y)
    t8 = bm.exp((self.q_half - z + self.IT_th) / self.q_k)
    q_inf_by_z = 1. / (1. + t8)  # q_{\infty}(z)

    # x
    gNa = self.g_Na * m_inf_by_V ** 3 * h_inf_by_y  # gNa
    gK = self.g_K * n_inf_by_y ** 4  # gK
    gT = self.g_T * p_inf_by_y * p_inf_by_y * q_inf_by_z  # gT
    FV = gNa + gK + gT + self.g_L + self.g_KL  # dF/dV
    Fm = 3 * self.g_Na * h_inf_by_y * (V - self.E_Na) * m_inf_by_V * m_inf_by_V * m_inf_by_V_diff  # dF/dvm
    t9 = self.C / m_tau_by_V
    t10 = FV + Fm
    t11 = t9 + FV
    rho_V = (t11 - bm.sqrt(bm.maximum(t11 ** 2 - 4 * t9 * t10, 0.))) / 2 / t10  # rho_V
    INa = gNa * (V - self.E_Na)
    IK = gK * (V - self.E_KL)
    IT = gT * (V - self.E_T)
    IL = self.g_L * (V - self.E_L)
    IKL = self.g_KL * (V - self.E_KL)
    Iext = self.V_factor * Isyn
    dVdt = rho_V * (-INa - IK - IT - IL - IKL + Iext) / self.C

    return dVdt

  def fy(self, y, t, V):
    # m channel
    t1 = 13. - V + self.NaK_th
    t1_exp = bm.exp(t1 / 4.)
    m_alpha_by_V = 0.32 * t1 / (t1_exp - 1.)  # \alpha_m(V)
    t2 = V - 40. - self.NaK_th
    t2_exp = bm.exp(t2 / 5.)
    m_beta_by_V = 0.28 * t2 / (t2_exp - 1.)  # \beta_m(V)
    m_inf_by_V = m_alpha_by_V / (m_alpha_by_V + m_beta_by_V)  # \m_{\infty}(V)

    # h channel
    h_alpha_by_V = 0.128 * bm.exp((17. - V + self.NaK_th) / 18.)  # \alpha_h(V)
    h_beta_by_V = 4. / (bm.exp((40. - V + self.NaK_th) / 5.) + 1.)  # \beta_h(V)
    h_inf_by_V = h_alpha_by_V / (h_alpha_by_V + h_beta_by_V)  # h_{\infty}(V)
    h_tau_by_V = 1. / self.phi_h / (h_alpha_by_V + h_beta_by_V)  # \tau_h(V)
    h_alpha_by_y = 0.128 * bm.exp((17. - y + self.NaK_th) / 18.)  # \alpha_h(y)
    t3 = bm.exp((40. - y + self.NaK_th) / 5.)
    h_beta_by_y = 4. / (t3 + 1.)  # \beta_h(y)
    h_beta_by_y_diff = 0.8 * t3 / (1 + t3) ** 2  # \beta_h'(y)
    h_inf_by_y = h_alpha_by_y / (h_alpha_by_y + h_beta_by_y)  # h_{\infty}(y)
    h_alpha_by_y_diff = - h_alpha_by_y / 18.  # \alpha_h'(y)
    h_inf_by_y_diff = (h_alpha_by_y_diff * h_beta_by_y - h_alpha_by_y * h_beta_by_y_diff) / \
                      (h_beta_by_y + h_alpha_by_y) ** 2  # h_{\infty}'(y)

    # n channel
    t4 = (15. - V + self.NaK_th)
    n_alpha_by_V = 0.032 * t4 / (bm.exp(t4 / 5.) - 1.)  # \alpha_n(V)
    n_beta_by_V = self.b * bm.exp((10. - V + self.NaK_th) / 40.)  # \beta_n(V)
    n_tau_by_V = 1. / (n_alpha_by_V + n_beta_by_V) / self.phi_n  # \tau_n(V)
    n_inf_by_V = n_alpha_by_V / (n_alpha_by_V + n_beta_by_V)  # n_{\infty}(V)
    t5 = (15. - y + self.NaK_th)
    t5_exp = bm.exp(t5 / 5.)
    n_alpha_by_y = 0.032 * t5 / (t5_exp - 1.)  # \alpha_n(y)
    t6 = bm.exp((10. - y + self.NaK_th) / 40.)
    n_beta_y = self.b * t6  # \beta_n(y)
    n_inf_by_y = n_alpha_by_y / (n_alpha_by_y + n_beta_y)  # n_{\infty}(y)
    n_alpha_by_y_diff = (0.0064 * t5 * t5_exp - 0.032 * (t5_exp - 1.)) / (t5_exp - 1.) ** 2  # \alpha_n'(y)
    n_beta_by_y_diff = -n_beta_y / 40  # \beta_n'(y)
    n_inf_by_y_diff = (n_alpha_by_y_diff * n_beta_y - n_alpha_by_y * n_beta_by_y_diff) / \
                      (n_alpha_by_y + n_beta_y) ** 2  # n_{\infty}'(y)

    # p channel
    p_inf_by_V = 1. / (1. + bm.exp((self.p_half - V + self.IT_th) / self.p_k))  # p_{\infty}(V)
    p_tau_by_V = (3 + 1. / (bm.exp((V + 27. - self.IT_th) / 10.) +
                            bm.exp(-(V + 102. - self.IT_th) / 15.))) / self.phi_p  # \tau_p(V)
    t7 = bm.exp((self.p_half - y + self.IT_th) / self.p_k)
    p_inf_by_y = 1. / (1. + t7)  # p_{\infty}(y)
    p_inf_by_y_diff = t7 / self.p_k / (1. + t7) ** 2  # p_{\infty}'(y)

    #  y
    Fvh = self.g_Na * m_inf_by_V ** 3 * (V - self.E_Na) * h_inf_by_y_diff  # dF/dvh
    Fvn = 4 * self.g_K * (V - self.E_KL) * n_inf_by_y ** 3 * n_inf_by_y_diff  # dF/dvn
    f4 = Fvh + Fvn
    rho_h = (1 - self.rho_p) * Fvh / f4
    rho_n = (1 - self.rho_p) * Fvn / f4
    fh = (h_inf_by_V - h_inf_by_y) / h_tau_by_V / h_inf_by_y_diff
    fn = (n_inf_by_V - n_inf_by_y) / n_tau_by_V / n_inf_by_y_diff
    fp = (p_inf_by_V - p_inf_by_y) / p_tau_by_V / p_inf_by_y_diff
    dydt = rho_h * fh + rho_n * fn + self.rho_p * fp

    return dydt

  def fz(self, z, t, V):
    q_inf_by_V = 1. / (1. + bm.exp((self.q_half - V + self.IT_th) / self.q_k))  # q_{\infty}(V)
    t8 = bm.exp((self.q_half - z + self.IT_th) / self.q_k)
    q_inf_by_z = 1. / (1. + t8)  # q_{\infty}(z)
    q_inf_diff_z = t8 / self.q_k / (1. + t8) ** 2  # q_{\infty}'(z)
    q_tau_by_V = (85. + 1 / (bm.exp((V + 48. - self.IT_th) / 4.) +
                             bm.exp(-(V + 407. - self.IT_th) / 50.))) / self.phi_q  # \tau_q(V)
    dzdt = (q_inf_by_V - q_inf_by_z) / q_tau_by_V / q_inf_diff_z
    return dzdt

  def derivative(self, V, y, z, t, Isyn):
    dvdt = self.fV(V, t, y, z, Isyn)
    dydt = self.fy(y, t, V)
    dzdt = self.fz(z, t, V)
    return dvdt, dydt, dzdt

  def update(self, t, dt):
    if isinstance(self.int_V, bp.ode.ExponentialEuler):
      V = self.int_V(self.V, t, self.y, self.z, self.input, dt=dt)
      self.y.value = self.int_y(self.y, t, self.V, dt=dt)
      self.z.value = self.int_z(self.z, t, self.V, dt=dt)
    else:
      V, self.y.value, self.z.value = self.integral(self.V, self.y, self.z, t, self.input, dt=dt)
    self.spike.value = bm.logical_and((self.V < self.Vth), (V >= self.Vth))
    self.V.value = V
    self.input[:] = 0.


def try1():
  trn = ReducedTRNModel(1, method='rk4')
  trn.b = 0.5
  trn.rho_p = 0.01

  # pp = bp.analysis.PhasePlane2D(
  #   [trn.int_V, trn.int_y],
  #   target_vars={'V': [-90., 70.], 'y': [-90., 70.]},
  #   pars_update={'z': -65., 'Isyn': 0.},
  #   resolutions=0.05,
  #   options={bp.analysis.C.y_by_x_in_fy: lambda V: V}
  # )
  # pp.plot_nullcline()
  # pp.plot_vector_field()
  # pp.plot_fixed_point(show=True)
  # # pp.plot_trajectory(initials={'V': [0.], 'y': [0.]}, duration=200.,
  # #                    plot_durations=[50, 100],
  # #                    dt=0.0001, show=True)

  pp = bp.analysis.Bifurcation2D(
    [trn.int_V, trn.int_y],
    target_vars={'V': [-100., 55.], 'y': [-100., 55.]},
    target_pars={'z': [-90., -40.]},
    pars_update={'Isyn': -0.05},
    resolutions=0.05,
    options={bp.analysis.C.y_by_x_in_fy: lambda V: V}
  )
  pp.plot_bifurcation()
  pp.plot_limit_cycle_by_sim(duration=100, dt=0.01, show=True)


  # pp = bp.analysis.FastSlow2D(
  #   [trn.int_V, trn.int_y, trn.int_z],
  #   fast_vars={'V': [-90., 55.], 'y': [-90., 55.]},
  #   slow_vars={'z': [-90., -50.]},
  #   pars_update={'Isyn': 0.},
  #   resolutions=0.05,
  #   options={bp.analysis.C.y_by_x_in_fy: lambda V: V}
  # )
  # pp.plot_bifurcation(show=True)


if __name__ == '__main__':
  try1()
  # try2()
