# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.numpy as np


def get_NaK_model(V_th=20., type='low-threshold'):
    """The I-Na,p+I-K model.

    Parameters
    ----------
    V_th : float
        A float number that specifies the threshold of treating
        elevated membrane potential as a spike. Unit in ``mV``.
        Default is 20.
    type : str
        A string that specifies whether the I-Na,p+I-K model has a high
        (``high-threshold``) or low threshold (``low-threshold``).
        Default is ``low-threshold``.

    Returns
    -------
    return_dict : bp.NeuType
        The necessary variables.
    """

    if not (type == 'low-threshold' or 'high-threshold'):
        raise ValueError("Argument `type` must be either `low-threshold`"
                         "or `high-threshold`")

    if type == 'high-threshold':
        C = 1
        E_L = -80
        g_L = 8
        g_Na = 20
        g_K = 10
        E_K = -90
        E_Na = 60
        Vm_half = -20
        k_m = 15
        Vn_half = -25
        k_n = 5
        tau = 1
    else:  # low-threshold
        C = 1
        E_L = -78  # different from high-threshold model
        g_L = 8
        g_Na = 20
        g_K = 10
        E_K = -90
        E_Na = 60
        Vm_half = -20
        k_m = 15
        Vn_half = -45  # different from high-threshold model
        k_n = 5
        tau = 1

    ST = bp.types.NeuState({'V': -65., 'n': 0., 'sp': 0., 'inp': 0.})

    @bp.integrate
    def int_n(n, t, V):
        n_inf = 1 / (1 + np.exp((Vn_half - V) / k_n))
        dndt = (n_inf - n) / tau
        return dndt

    @bp.integrate
    def int_V(V, t, n, input):
        m_inf = 1 / (1 + np.exp((Vm_half - V) / k_m))
        I_leak = g_L * (V - E_L)
        I_Na = g_Na * m_inf * (V - E_Na)
        I_K = g_K * n * (V - E_K)
        dvdt = (-I_leak - I_Na - I_K + input) / C
        return dvdt

    def update(ST, _t_):
        n = np.clip(int_n(ST['n'], _t_, ST['V']), 0., 1.)
        V = int_V(ST['V'], _t_, n, ST['inp'])
        sp = np.logical_and(ST['V'] < V_th, V >= V_th)
        ST['V'] = V
        ST['n'] = n
        ST['sp'] = sp
        ST['inp'] = 0.

    return bp.NeuType(name="NaK_model",
                      requires=dict(ST=ST),
                      steps=update,
                      vector_based=True)


analyzer = bp.BifurcationAnalyzer(
    model=get_NaK_model(),
    target_pars={'input': [0, 50.]},
    dynamical_vars={"V": [-90., 20.], 'n': [0., 1.]},
    par_resolution=0.1)

analyzer.plot_bifurcation(plot_vars='V', show=True)
