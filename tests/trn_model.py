# -*- coding: utf-8 -*-

import numpy as np

import brainpy as bp


def get_TRN_reduced(rho_p=0.99, IL=dict(), IKL=dict()):
    # parameters of INa
    g_Na, E_Na, shift_NaK = 100., 50., -55.
    phi_m, m_half, m_k = 1., 25.97463508, 7.412050705
    phi_h, h_half, h_k = 1., 21.680489665, -4.0405937

    # parameters of IK
    g_K, E_K = 10., -95.
    phi_n, b = 1., 0.5 * 0.28

    # parameters of IT
    g_Ca, E_Ca, shift_IT = 2., 120., -3.
    phi_p, p_half, p_k = 6.9, -52., 7.4
    phi_q, q_half, q_k = 3.7, -80., -5.

    # parameters of IL
    g_L = IL.pop('g_L', 0.02)
    E_L = IL.pop('E_L', -70.)

    # parameters of IKL
    g_KL = IKL.pop('g_KL', 0.005)
    E_KL = IKL.pop('E_KL', -95.)

    # parameters of V
    C, Vth, area = 1., 0., 1.43e-4
    V_factor = 1e-3 / area

    @bp.integrate
    def int_V(V, t, vy, vz, Isyn):
        # m channel
        a1_exp = np.exp((m_half - V + shift_NaK) / m_k)
        m_inf_x = 1. / (1. + a1_exp)
        m_inf_x_twice = m_inf_x ** 2
        m_inf_x_cubic = m_inf_x ** 3
        m_inf_diff_x = a1_exp / m_k / (1. + a1_exp) ** 2
        a2 = 13. - V + shift_NaK
        m_alpha = 0.32 * a2 / (np.exp(a2 / 4.) - 1.)
        a3 = V - 40. - shift_NaK
        m_beta = 0.28 * a3 / (np.exp(a3 / 5.) - 1.)
        m_tau = 1. / phi_m / (m_alpha + m_beta)

        # gNa
        h_inf_y = 1. / (1. + np.exp((h_half - vy + shift_NaK) / h_k))
        gNa = g_Na * m_inf_x_cubic * h_inf_y

        # gK
        c4 = (15. - vy + shift_NaK)
        n_alpha_y = 0.032 * c4 / (np.exp(c4 / 5.) - 1.)
        n_beta_y = b * np.exp((10. - vy + shift_NaK) / 40.)
        n_inf_y = n_alpha_y / (n_alpha_y + n_beta_y)
        n_inf_y_fourtimes = n_inf_y ** 4
        gK = g_K * n_inf_y_fourtimes

        # gT
        p_inf_y = 1. / (1. + np.exp((p_half - vy + shift_IT) / p_k))
        p_inf_y_square = p_inf_y ** 2
        q_inf_z = 1. / (1. + np.exp((q_half - vz + shift_IT) / q_k))
        gT = g_Ca * p_inf_y_square * q_inf_z

        # dF/dV
        FV = gNa + gK + gT + g_L + g_KL

        # dF/dvm
        Fvm = 3 * g_Na * h_inf_y * (V - E_Na) * m_inf_x_twice * m_inf_diff_x

        # rho_V
        f1 = C / m_tau
        f2 = FV + Fvm
        f3 = f1 + FV
        rho_V = (f3 - np.sqrt(f3 ** 2 - 4 * f1 * f2)) / 2 / f2

        # currents
        INa = gNa * (V - E_Na)
        IK = gK * (V - E_K)
        IT = gT * (V - E_Ca)
        IL = g_L * (V - E_L)
        IKL = g_KL * (V - E_KL)
        Iext = V_factor * Isyn

        # dvdt
        dxdt = rho_V * (-INa - IK - IT - IL - IKL + Iext) / C

        return dxdt

    @bp.integrate
    def int_vy(vy, t, V):
        # h channel
        b1_exp = np.exp((h_half - V + shift_NaK) / h_k)
        h_inf_x = 1. / (1. + b1_exp)
        b2_exp = np.exp((h_half - vy + shift_NaK) / h_k)
        h_inf_y = 1. / (1. + b2_exp)
        h_inf_diff_y = b2_exp / h_k / (1. + b2_exp) ** 2
        h_alpha = 0.128 * np.exp((17. - V + shift_NaK) / 18.)
        h_beta = 4. / (np.exp((40. - V + shift_NaK) / 5.) + 1.)
        h_tau = 1. / phi_h / (h_alpha + h_beta)

        # n channel
        c3 = (15. - V + shift_NaK)
        n_alpha_x = 0.032 * c3 / (np.exp(c3 / 5.) - 1.)
        n_beta_x = b * np.exp((10. - V + shift_NaK) / 40.)
        n_tau_x = 1. / (n_alpha_x + n_beta_x) / phi_n
        n_inf_x = n_alpha_x / (n_alpha_x + n_beta_x)

        c4 = (15. - vy + shift_NaK)
        c4_1 = c4 / 5.
        c4_exp = np.exp(c4_1)
        n_alpha_y = 0.032 * c4 / (c4_exp - 1.)
        c5_exp = np.exp((10. - vy + shift_NaK) / 40.)
        n_beta_y = b * c5_exp
        n_inf_y = n_alpha_y / (n_alpha_y + n_beta_y)
        n_alpha_y_diff = (0.032 * c4_1 * c4_exp - 0.032 * (c4_exp - 1.)) / (c4_exp - 1.) ** 2
        n_beta_y_diff = -b / 40 * c5_exp
        n_inf_diff_y = (n_alpha_y_diff * n_beta_y - n_alpha_y * n_beta_y_diff) / (n_alpha_y + n_beta_y) ** 2

        # p channel
        d1_exp = np.exp((p_half - V + shift_IT) / p_k)
        p_inf_x = 1. / (1. + d1_exp)
        d2_exp = np.exp((p_half - vy + shift_IT) / p_k)
        p_inf_y = 1. / (1. + d2_exp)
        p_inf_diff_y = d2_exp / p_k / (1. + d2_exp) ** 2
        d3 = np.exp((V + 27. - shift_IT) / 10.)
        d4 = np.exp(-(V + 102. - shift_IT) / 15.)
        p_tau = (3 + 1. / (d3 + d4)) / phi_p

        # dF/dvh, dF/dvn
        m_inf_x = 1. / (1. + np.exp((m_half - V + shift_NaK) / m_k))
        m_inf_x_cubic = m_inf_x ** 3
        Fvh = g_Na * m_inf_x_cubic * (V - E_Na) * h_inf_diff_y
        Fvn = 4 * g_K * (V - E_K) * n_inf_y ** 3 * n_inf_diff_y

        # reduction coefficients
        f4 = Fvh + Fvn
        rho_h = (1 - rho_p) * Fvh / f4
        rho_n = (1 - rho_p) * Fvn / f4

        # function values
        fh = (h_inf_x - h_inf_y) / h_tau / h_inf_diff_y
        fn = (n_inf_x - n_inf_y) / n_tau_x / n_inf_diff_y
        fp = (p_inf_x - p_inf_y) / p_tau / p_inf_diff_y

        # dydt
        dydt = rho_h * fh + rho_n * fn + rho_p * fp
        return dydt

    @bp.integrate
    def int_vz(vz, t, V):
        # q channel
        e1_exp = np.exp((q_half - V + shift_IT) / q_k)
        q_inf_x = 1. / (1. + e1_exp)
        e2_exp = np.exp((q_half - vz + shift_IT) / q_k)
        q_inf_z = 1. / (1. + e2_exp)
        q_inf_diff_z = e2_exp / q_k / (1. + e2_exp) ** 2
        e2 = np.exp((V + 48. - shift_IT) / 4.)
        e3 = np.exp(-(V + 407. - shift_IT) / 50.)
        q_tau = (85. + 1 / (e2 + e3)) / phi_q
        # dzdt
        dzdt = (q_inf_x - q_inf_z) / q_tau / q_inf_diff_z
        return dzdt

    def update_state(ST, _t):
        vy = int_vy(ST['vy'], _t, ST['V'])
        vz = int_vz(ST['vz'], _t, ST['V'])
        V = int_V(ST['V'], _t, vy, vz, ST['input'])
        sp = np.logical_and(ST['V'] < Vth, V >= Vth)
        ST['V'] = V
        ST['vy'] = vy
        ST['vz'] = vz
        ST['spike'] = sp
        ST['input'] = 0.

    def init_state(neu_state, Vr):
        neu_state['V'] = Vr
        neu_state['vy'] = neu_state['V'].copy()
        neu_state['vz'] = neu_state['V'].copy()

    TRN = bp.NeuType(name='TRN',
                     ST=bp.types.NeuState(['V', 'vy', 'vz', 'spike', 'input']),
                     steps=update_state,
                     mode='vector',
                     hand_overs={'init_state': init_state})

    return TRN


trn = get_TRN_reduced(rho_p=0.8, IL=dict(g_L=0.05, E_L=-77.),
                      IKL=dict(g_KL=0.00792954, E_KL=-95.))

analyzer = bp.PhasePortraitAnalyzer(
    model=trn,
    target_vars={'V': [-100., 80.], 'vy': [-100., 50.]},
    fixed_vars={'vz': -75., 'Isyn': 0.},
    options={'escape_sympy_parser': True}
)
analyzer.plot_nullcline(resolution=0.07)
analyzer.plot_vector_field()
analyzer.plot_fixed_point(show=True)
# analyzer.plot_trajectory([(-75, 75, 100.), (-10.5, 10.4, 100.)],
#                          show=True)
#
