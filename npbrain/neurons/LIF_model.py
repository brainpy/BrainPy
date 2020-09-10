# -*- coding: utf-8 -*-

from ..core import integrate
from ..core.neuron_group import *
from ..utils import autojit

__all__ = [
    'LIF'
]


def LIF(geometry, method=None, tau=10., Vr=0., Vth=10., noise=0., ref=0., name='LIF'):
    """Leaky integrate-and-fire neuron model.

    Parameters
    ----------
    geometry : int, list, tuple
        The geometry of neuron group. If an integer is given, it is the size
        of the population.
    method : str, callable, dict
        The numerical integration method. Either a string with the name of a
        registered method (e.g. "euler") or a function.
    tau : float
        Neuron parameters.
    Vr : float
        The reset potential.
    Vth : float
        The spike threshold.
    noise : float, callable
        The noise item.
    ref : float
        The refractory period.
    name : str
        The name of the neuron group.

    Returns
    -------
    neurons : Neurons
        The created neuron group.
    """

    var2index = {'V': 0}
    num, geometry = format_geometry(geometry)
    state = init_neu_state(num, [('V', Vr)])

    judge_spike = get_spike_judger()

    @integrate(method=method, noise=noise / tau, signature='f[:](f[:], f, f[:])')
    def int_f(V, t, Isyn):
        return (-V + Vr + Isyn) / tau

    if ref > 0.:

        @autojit('void(f[:, :], f)')
        def update_state(neu_state, t):
            V_new = int_f(neu_state[0], t, neu_state[-1])
            for idx in range(num):
                if (t - neu_state[-2, idx]) > ref:
                    v = V_new[idx]
                    if v >= Vth:
                        neu_state[-5, idx] = 0.  # refractory state
                        neu_state[-3, idx] = 1.  # spike state
                        neu_state[-2, idx] = t  # spike time
                        v = Vr
                    else:
                        neu_state[-5, idx] = 1.
                        neu_state[-3, idx] = 0.
                    neu_state[0, idx] = v  # membrane potential
                else:
                    neu_state[-5, idx] = 0.
                    neu_state[-3, idx] = 0.

    else:

        @autojit('void(f[:, :], f)')
        def update_state(neu_state, t):
            neu_state[0] = int_f(neu_state[0], t, neu_state[-1])
            spike_idx = judge_spike(neu_state, Vth, t)
            neu_state[0][spike_idx] = Vr

    return Neurons(**locals())
