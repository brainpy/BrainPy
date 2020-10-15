# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import npbrain as nb
from npbrain import numpy as np

nb.profile.set_backend('numba')
nb.profile.show_formatted_code = True
nb.profile.merge_integral = True


def define_ampa1_single(g_max=0.10, E=0., tau_decay=2.0):
    """AMPA conductance-based synapse (type 1).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{d s}{d t}&=-\\frac{s}{\\tau_{decay}}+\\sum_{k} \\delta(t-t_{j}^{k})

    Parameters
    ----------
    g_max : float
        Maximum conductance.
    E : float
        Reversal potential.
    tau_decay : float
        Tau for decay.
    """

    requires = dict(
        ST=nb.types.SynState(['s'], help='AMPA synapse state.'),
        pre=nb.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "sp" item.'),
        post=nb.types.NeuState(['V', 'inp'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
    )

    @nb.integrate(method='euler')
    def ints(s, t):
        return - s / tau_decay

    @nb.delay_push
    def update(ST, _t_, pre):
        s = ints(ST['s'], _t_)
        s += pre['sp']
        ST['s'] = s

    @nb.delay_pull
    def output(ST, post):
        post_val = - g_max * ST['s'] * (post['V'] - E)
        post['inp'] += post_val

    return {'requires': requires, 'steps': (update, output)}


AMPA1_single = nb.SynType(name='AMPA_type1', create_func=define_ampa1_single, vector_based=False)


def define_ampa2_single(g_max=0.42, E=0., alpha=0.98, beta=0.18, T=0.5, T_duration=0.5):
    """AMPA conductance-based synapse (type 2).

    .. math::

        I_{syn}&=\\bar{g}_{syn} s (V-E_{syn})

        \\frac{ds}{dt} &=\\alpha[T](1-s)-\\beta s

    Parameters
    ----------
    g_max : float
        Maximum conductance.
    E : float
        Reversal potential.
    alpha
    beta
    T
    T_duration
    """

    requires = {
        'ST': nb.types.SynState({'s': 0., 'sp_t': -1e7},
                                help="""
                                    "s": Synaptic state.
                                    "sp_t": Pre-synaptic neuron spike time.
                                """),
        'pre': nb.types.NeuState(['sp'], help='Pre-synaptic neuron state must have "sp" item.'),
        'post': nb.types.NeuState(['V', 'inp'], help='Pre-synaptic neuron state must have "V" and "inp" item.'),
    }

    @nb.integrate(method='euler')
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    @nb.delay_push
    def update(ST, _t_, pre):
        if pre['sp'] > 0.:
            ST['sp_t'] = _t_
        TT = ((_t_ - ST['sp_t']) < T_duration) * T
        s = np.clip(int_s(ST['s'], _t_, TT), 0., 1.)
        ST['s'] = s

    @nb.delay_pull
    def output(ST, post):
        post_val = - g_max * ST['s'] * (post['V'] - E)
        post['inp'] += post_val

    return {'requires': requires, 'steps': (update, output)}


AMPA2_single = nb.SynType(name='AMPA_type2', create_func=define_ampa2_single, vector_based=False)


def run_ampa_single(cls, duration=650.):
    from examples.neurons.HH_model import HH
    nb.profile.set_backend('numba')
    nb.profile.show_formatted_code = True

    pre = nb.NeuGroup(HH, 10)
    post = nb.NeuGroup(HH, 20)
    ampa = nb.SynConn(model=cls, pre_group=pre, post_group=post, conn=nb.connect.All2All(),
                      monitors=['s'], delay=10.)

    net = nb.Network(pre, ampa, post)
    Iext = nb.inputs.spike_current([10, 110, 210, 310, 410], nb.profile._dt, 1., duration=duration)
    net.run(duration, inputs=(ampa, 'pre.sp', Iext, '='), report=True)

    fig, gs = nb.visualize.get_figure(1, 1, 5, 10)
    fig.add_subplot(gs[0, 0])
    plt.plot(net.ts, ampa.mon.s[:, 0], label='s')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_ampa_single(AMPA1_single)
    # run_ampa_single(AMPA2_single)
