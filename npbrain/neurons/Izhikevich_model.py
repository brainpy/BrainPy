# -*- coding: utf-8 -*-

from .. import _numpy as np

from npbrain.core import integrate
from npbrain.core.neuron_group import *
from ..utils import autojit

__all__ = [
    'Izhikevich'
]


def Izhikevich(geometry, mode=None, method=None, a=0.02, b=0.20, c=-65., d=8.,
               ref=0., noise=0., Vth=30., Vr=-65., name='Izhikevich_neuron'):
    """Izhikevich two-variable neuron model.

    Parameters
    ----------
    mode : None, str
        At least twenty firing modes have beed provides by Izhikevich.
        One can specify the preferred firing mode to get the corresponding
        neuron group.
    geometry : int, a_list, tuple
        The geometry of neuron group. If an integer is given, it is the size
        of the population.
    method : str, callable, dict
        The numerical integration method. Either a string with the name of a
        registered method (e.g. "euler") or a function.
    a : float
        It determines the time scale of the recovery variable :math:`u`.
    b : float
        It describes the sensitivity of the recovery variable :math:`u` to
        the sub-threshold fluctuations of the membrane potential :math:`v`.
    c : float
        It describes the after-spike reset value of the membrane potential
        :math:`v` caused by the fast high-threshold :math:`K^{+}` conductance.
    d : float
        It describes after-spike reset of the recovery variable :math:`u` caused
        by slow high-threshold :math:`Na^{+}` and :math:`K^{+}` conductance.
    ref
    noise
    Vth
    Vr
    name : str
        The name of the neuron group.

    Returns
    -------
    neurons : Neurons
        The created neuron group.
    """

    var2index = {'V': 0, 'u': 1}
    num, geometry = format_geometry(geometry)
    state = init_neu_state(num_neu=num, variables=len(var2index))

    if mode in ['tonic', 'tonic spiking']:
        a, b, c, d = [0.02, 0.40, -65.0, 2.0]
    elif mode in ['phasic', 'phasic spiking']:
        a, b, c, d = [0.02, 0.25, -65.0, 6.0]
    elif mode in ['tonic bursting']:
        a, b, c, d = [0.02, 0.20, -50.0, 2.0]
    elif mode in ['phasic bursting']:
        a, b, c, d = [0.02, 0.25, -55.0, 0.05]
    elif mode in ['mixed mode']:
        a, b, c, d = [0.02, 0.20, -55.0, 4.0]
    elif mode in ['SFA', 'spike frequency adaptation']:
        a, b, c, d = [0.01, 0.20, -65.0, 8.0]
    elif mode in ['Class 1', 'class 1']:
        a, b, c, d = [0.02, -0.1, -55.0, 6.0]
    elif mode in ['Class 2', 'class 2']:
        a, b, c, d = [0.20, 0.26, -65.0, 0.0]
    elif mode in ['spike latency', ]:
        a, b, c, d = [0.02, 0.20, -65.0, 6.0]
    elif mode in ['subthreshold oscillation', ]:
        a, b, c, d = [0.05, 0.26, -60.0, 0.0]
    elif mode in ['resonator', ]:
        a, b, c, d = [0.10, 0.26, -60.0, -1.0]
    elif mode in ['integrator', ]:
        a, b, c, d = [0.02, -0.1, -55.0, 6.0]
    elif mode in ['rebound spike', ]:
        a, b, c, d = [0.03, 0.25, -60.0, 4.0]
    elif mode in ['rebound burst', ]:
        a, b, c, d = [0.03, 0.25, -52.0, 0.0]
    elif mode in ['threshold variability', ]:
        a, b, c, d = [0.03, 0.25, -60.0, 4.0]
    elif mode in ['bistability', ]:
        a, b, c, d = [1.00, 1.50, -60.0, 0.0]
    elif mode in ['DAP', 'depolarizing afterpotential']:
        a, b, c, d = [1.00, 0.20, -60.0, -21.0]
    elif mode in ['accomodation', ]:
        a, b, c, d = [0.02, 1.00, -55.0, 4.0]
    elif mode in ['inhibition-induced spiking', ]:
        a, b, c, d = [-0.02, -1.00, -60.0, 8.0]
    elif mode in ['inhibition-induced bursting', ]:
        a, b, c, d = [-0.026, -1.00, -45.0, 0.0]

    def init_state(state_, Vr_):
        state_[0] = np.ones(num) * Vr_
        state_[1] = state_[0] * b

    init_state(state, Vr)
    judge_spike = get_spike_judger()

    @integrate(method=method, signature='f[:](f[:], f, f[:])')
    def int_u(u, t, V):
        return a * (b * V - u)

    @integrate(method=method, noise=noise, signature='f[:](f[:], f, f[:], f[:])')
    def int_V(V, t, u, Isyn):
        return 0.04 * V * V + 5 * V + 140 - u + Isyn

    if ref > 0.:

        @autojit('void(f[:, :], f)')
        def update_state(neu_state, t):
            not_ref = (t - neu_state[-2]) > ref
            V, u, Isyn = neu_state[0], neu_state[1], neu_state[-1]
            u_new = int_u(u, t, V)
            V_new = int_V(V, t, u, Isyn)
            not_ref_idx = np.where(not_ref)[0]
            for idx in not_ref_idx:
                neu_state[0, idx] = V_new[idx]
                neu_state[1, idx] = u_new[idx]
            spike_idx = judge_spike(neu_state, Vth, t)
            for idx in spike_idx:
                neu_state[0, idx] = c
                neu_state[1, idx] += d
                neu_state[-5, idx] = 0.
    else:

        @autojit('void(f[:, :], f)')
        def update_state(neu_state, t):
            V, u, Isyn = neu_state[0], neu_state[1], neu_state[-1]
            neu_state[0] = int_V(V, t, u, Isyn)
            neu_state[1] = int_u(u, t, V)
            spike_idx = judge_spike(neu_state, Vth, t)
            for idx in spike_idx:
                neu_state[0, idx] = c
                neu_state[1, idx] += d

    return Neurons(**locals())
