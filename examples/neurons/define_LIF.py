# -*- coding: utf-8 -*-

import npbrain as nn


def LIF(geometry, tau=100, V_reset=0., Vth=1.):
    var2index = dict(V=0)
    num, geometry = nn.format_geometry(geometry)

    state = nn.initial_neu_state(1, num)
    state[0] = V_reset

    @nn.integrate
    def int_f(V, t, Isyn):
        return (-V + Isyn) / tau

    def update_state(neu_state, t):
        V_new = int_f(neu_state[0], t, neu_state[-1])
        neu_state[0] = V_new
        spike_idx = nn.judge_spike(neu_state, Vth, t)
        neu_state[0][spike_idx] = V_reset

    return nn.Neurons(**locals())
