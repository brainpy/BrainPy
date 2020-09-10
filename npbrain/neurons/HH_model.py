# -*- coding: utf-8 -*-

from .. import _numpy as np

from ..core import integrate
from ..core.neuron_group import *
from ..utils import get_clip
from ..utils import autojit

__all__ = [
    'HH'
]


def HH(geometry, method=None, noise=0., E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387,
       g_Leak=0.03, C=1.0, Vr=-65., Vth=20., name='Hodgkin–Huxley_neuron'):
    """The Hodgkin–Huxley neuron model.

    The Hodgkin–Huxley model can be thought of as a differential equation
    with four state variables, :math:`v(t)`, :math:`m(t)`, :math:`n(t)`, and
    :math:`h(t)`, that change with respect to time :math:`t`.

    Parameters
    ----------
    geometry : int, list, tuple
        The geometry of neuron group. If an integer is given, it is the size
        of the population.
    method : str, callable, dict
        The numerical integration method. Either a string with the name of a
        registered method (e.g. "euler") or a function.
    noise
    E_Na
    g_Na
    E_K
    g_K
    E_Leak
    g_Leak
    C
    Vr
    Vth
    name

    Returns
    -------
    neurons : Neurons
        The created neuron group.
    """

    var2index = {'V': 0, 'm': 1, 'h': 2, 'n': 3}
    num, geometry = format_geometry(geometry)
    state = init_neu_state(num, variables=len(var2index))

    def init_state(neu_state, Vr_):
        V = np.ones(num) * Vr_
        neu_state[0] = V  # V
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        neu_state[1] = alpha / (alpha + beta)  # m
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        neu_state[2] = alpha / (alpha + beta)  # h
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        neu_state[3] = alpha / (alpha + beta)  # n

    init_state(state, Vr)
    judge_spike = get_spike_judger()
    clip = get_clip()

    @integrate(method=method, signature='f[:](f[:], f, f[:])')
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m

    @integrate(method=method, signature='f[:](f[:], f, f[:])')
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h

    @integrate(method=method, signature='f[:](f[:], f, f[:])')
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n

    @integrate(method=method, noise=noise / C,
               signature='f[:](f[:], f, f[:], f[:], f[:], f[:])')
    def int_V(V, t, m, h, n, Isyn):
        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + Isyn) / C
        return dvdt

    @autojit('void(f[:, :], f)')
    def update_state(neu_state, t):
        V, Isyn = neu_state[0], neu_state[-1]
        m = clip(int_m(neu_state[1], t, V), 0., 1.)
        h = clip(int_h(neu_state[2], t, V), 0., 1.)
        n = clip(int_n(neu_state[3], t, V), 0., 1.)
        V = int_V(V, t, m, h, n, Isyn)
        neu_state[0] = V
        neu_state[1] = m
        neu_state[2] = h
        neu_state[3] = n
        judge_spike(neu_state, Vth, t)

    return Neurons(**locals())
