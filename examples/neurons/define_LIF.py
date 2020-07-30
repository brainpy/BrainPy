# -*- coding: utf-8 -*-

import numpy as np
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


def LIF_with_ref(geometry, tau=100, Vr=0., Vth=1., ref=1.):
    var2index = dict(V=0)
    num, geometry = nn.format_geometry(geometry)

    state = nn.initial_neu_state(1, num)
    state[0] = Vr

    @nn.integrate
    def int_f(V, t, Isyn):
        return (-V + Isyn) / tau

    def update_state(neu_state, t):
        # get variable
        V_old = neu_state[0]
        Isyn = neu_state[-1]
        last_sp_time = neu_state[-2]

        # calculate states
        not_in_ref = (t - last_sp_time) > ref
        V_new = int_f(V_old, t, Isyn)
        for idx in range(num):
            if not_in_ref[idx]:
                neu_state[0, idx] = V_new[idx]

        # judge spike
        spike_idx = nn.judge_spike(neu_state, Vth, t)
        neu_state[0][spike_idx] = Vr
        not_in_ref = 0.
        neu_state[-5] = not_in_ref

    return nn.Neurons(**locals())

