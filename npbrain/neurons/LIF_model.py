# -*- coding: utf-8 -*-

import numpy as np

from npbrain.core import integrate
from npbrain.core.neuron import *

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

    state = initial_neu_state(1, num)
    state[0] = Vr

    judge_spike = get_spike_judger()

    @integrate(method=method, noise=noise / tau)
    def int_f(V, t, Isyn):
        return (-V + Vr + Isyn) / tau

    if ref > 0.:
        def update_state(neu_state, t):
            in_ref = (t - neu_state[-2]) <= ref
            neu_state[-5] = in_ref
            # calculate states
            V = int_f(neu_state[0], t, neu_state[-1])
            # reset neuron values in refractory period
            in_ref_idx = np.where(in_ref)[0]
            V[in_ref_idx] = neu_state[0][in_ref_idx]
            neu_state[0] = V
            # get spikes
            spike_idx = judge_spike(neu_state, Vth, t)
            neu_state[0][spike_idx] = Vr
            neu_state[-5][spike_idx] = 1.
    else:
        def update_state(neu_state, t):
            neu_state[0] = int_f(neu_state[0], t, neu_state[-1])
            spike_idx = judge_spike(neu_state, Vth, t)
            neu_state[0][spike_idx] = Vr

    return Neurons(**locals())
