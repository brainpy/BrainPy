# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import brainpy as bp
import numpy as np


def define_LIF(tau=10., Vr=0., Vth=10., noise=0., ref=0.):
    """Leaky integrate-and-fire neuron model.

    Parameters
    ----------
    tau : float
        Membrane time constants.
    Vr : float
        The reset potential.
    Vth : float
        The spike threshold.
    noise : float, callable
        The noise item.
    ref : float
        The refractory period.
    """

    ST = bp.types.NeuState(
        {'V': 0, 'sp_t': -1e7, 'spike': 0., 'input': 0.},
    )

    @bp.integrate
    def int_f(V, t, Isyn):
        return (-V + Vr + Isyn) / tau, noise / tau

    def update(ST, _t):
        if _t - ST['sp_t'] > ref:
            V = int_f(ST['V'], _t, ST['input'])
            if V >= Vth:
                V = Vr
                ST['sp_t'] = _t
                ST['spike'] = True
            ST['V'] = V
        else:
            ST['spike'] = False
        ST['input'] = 0.

    return bp.NeuType(name='LIF', 
                      ST=ST,
                      steps=update, 
                      mode='scalar')


if __name__ == '__main__':
    bp.profile.set(jit=True, dt=0.02)

    LIF = define_LIF(noise=0., ref=5.)

    neu = bp.NeuGroup(LIF, geometry=(1,), monitors=['spike', 'V'])
    Iext, duration = bp.inputs.constant_current([(0, 20), (30, 50), (0, 30)])
    neu.run(duration=duration, inputs=['ST.inp', Iext], report=True)

    fig, gs = bp.visualize.get_figure(1, 1, 4, 6)
    fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(neu.mon.ts, neu.mon.V, ylabel='V', show=True)


if __name__ == '__main__1':
    bp.profile.set(jit=True, dt=0.02)

    LIF = define_LIF(noise=1., ref=5.)

    neu = bp.NeuGroup(LIF, geometry=(10,), monitors=['spike', 'V'])
    neu.pars['Vr'] = np.random.randint(0, 2, size=(10,))
    neu.pars['tau'] = np.random.randint(5, 10, size=(10,))
    neu.run(duration=100., inputs=['ST.inp', 13.], report=True)

    fig, gs = bp.visualize.get_figure(1, 1, 4, 8)

    fig.add_subplot(gs[0, 0])
    plt.plot(neu.mon.ts, neu.mon.V[:, 0], label=f'N-0 (tau={neu.pars.get("tau")[0]})')
    plt.plot(neu.mon.ts, neu.mon.V[:, 2], label=f'N-2 (tau={neu.pars.get("tau")[2]})')
    plt.ylabel('Membrane potential')
    plt.xlim(- 0.1, 100.1)
    plt.legend()
    plt.xlabel('Time (ms)')

    plt.show()
