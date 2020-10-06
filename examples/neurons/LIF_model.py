# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import npbrain as nb
import npbrain._numpy as np

nb.profile.set_backend('numpy')
nb.profile.set_dt(0.02)
nb.profile._show_codgen = True


def LIF_single_neuron(tau=10., Vr=0., Vth=10., noise=0., ref=0.):
    """Leaky integrate-and-fire neuron model.

    Parameters
    ----------
    tau : float
        NeuGroup parameters.
    Vr : float
        The reset potential.
    Vth : float
        The spike threshold.
    noise : float, callable
        The noise item.
    ref : float
        The refractory period.
    """

    ST = nb.types.NeuState({'V': 0, 'sp_t': -1e7, 'sp': 0., 'inp': 0.})

    @nb.integrate(noise=noise / tau)
    def int_f(V, t, Isyn):
        return (-V + Vr + Isyn) / tau

    def update(ST, _t_):
        if _t_ - ST['sp_t'] > ref:
            V = int_f(ST['V'], _t_, ST['inp'])
            if V >= Vth:
                V = Vr
                ST['sp_t'] = _t_
                ST['sp'] = True
            ST['V'] = V
        else:
            ST['sp'] = False
        ST['inp'] = 0.

    return {'requires': {'ST': ST}, 'steps': update}


LIF_single = nb.NeuType(name='LIF_neuron', create_func=LIF_single_neuron, group_based=False)


if __name__ == '__main__':
    neu = nb.NeuGroup(LIF_single, geometry=(10,), monitors=['sp', 'V'],
                      pars_update={'tau': np.random.randint(5, 10, size=(10,))})
    net = nb.Network(neu)
    net.run(duration=100., inputs=[neu, 'ST.inp', 13.], report=True)

    ts = net.ts
    fig, gs = nb.visualize.get_figure(1, 1, 4, 8)

    fig.add_subplot(gs[0, 0])
    plt.plot(ts, neu.mon.V[:, 0], label=f'N-0 (tau={neu.pars_update["tau"][0]})')
    plt.plot(ts, neu.mon.V[:, 2], label=f'N-2 (tau={neu.pars_update["tau"][2]})')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.legend()
    plt.xlabel('Time (ms)')

    plt.show()