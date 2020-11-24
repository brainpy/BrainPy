# -*- coding: utf-8 -*-

r"""
I-Na,p+I-K Model - a two-dimensional system model

$$ C\dot{V} = I - g_L * (V-E_L)-g_Na*m_\infinity(V)(V-E_Na)-g_K*n*(V-E_K) $$
$$ \dot{n} = (n_\infinity(V)-n)/\tau(V) $$
$$ m_\infinity(V) = 1 / (1+\exp{(V_halfm-V)/k_m}) $$
$$ n_\infinity(V) = 1 / (1+\exp{(V_halfn-V)/k_n}) $$

This model specifies a leak current I_L, persistent sodium current I_Na,p 
with instantaneous activation kinetic, and a relatively slower persistent 
potassium current I_K with either high or low threshold (the two choices 
result in fundamentally different dynamics).

Reference: Izhikevich, Eugene M. Dynamical systems in neuroscience (Chapter 
4). MIT press, 2007. 

Author: Lin Xiaohan (Oct. 9th, 2020)
"""

import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.numpy as np
from brainpy.dynamics import PhasePortraitAnalyzer2D

bp.profile.set_dt(0.02)
bp.profile.merge_integral = False


def define_NaK_model(V_th=20., type='low-threshold'):
    r"""The I-Na,p+I-K model.

    Parameters
    ----------
    noise
    V_th (float): a float number that specifies the threshold of treating 
            elevated membrane potential as a spike. Unit in ``mV``. Default 
            is 20.
    type (str): a string that specifies whether the I-Na,p+I-K model has a high
            (``high-threshold``) or low threshold (``low-threshold``). Default
            is ``low-threshold``.

    Returns
    -------
    return_dict : dict
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

    ST = bp.types.NeuState(
        {'V': -65., 'n': 0., 'sp': 0., 'inp': 0.},
        help='I-Na,p+I-K model neuron state.\n'
             '"V" denotes membrane potential.\n'
             '"n" denotes potassium channel activation probability.\n'
             '"sp" denotes spiking state.\n'
             '"inp" denotes synaptic input.\n'
    )

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


NaK_neuron = define_NaK_model()


group = bp.NeuGroup(NaK_neuron, 1, monitors=['V'])
group.run(50., inputs=('ST.inp', 50.))

bp.visualize.line_plot(group.mon.ts, group.mon.V, ylabel='Potential (mV)', show=True)


da = PhasePortraitAnalyzer2D(
    neuron=NaK_neuron,
    plot_variables=["n", "V"])
da.plot_nullcline(var_lim={'n': [0, 0.7],
                  'V':[-90, 20]},
                  sub_dict={"input": 50.})
da.plot_vector_field(inherit=True)
da.find_fixed_point(inherit=True)
plt.show()
