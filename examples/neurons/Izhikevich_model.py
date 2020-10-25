# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.numpy as np


def define_Izhikevich(a=0.02, b=0.20, c=-65., d=8., ref=0., noise=0., Vth=30., mode=None):
    """Izhikevich two-variable neuron model.

    Parameters
    ----------
    mode : optional, str
        The neuron spiking mode.
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
    ref : float
        Refractory period length. [ms]
    noise : float
        The noise fluctuation.
    Vth : float
        The membrane potential threshold.
    Vr : float
        The membrane reset potential.
    """

    state = bp.types.NeuState(
        {'V': 0., 'u': 1., 'sp': 0., 'sp_t': -1e7, 'inp': 0.},
        help='''
        Izhikevich two-variable neuron model state.
        
        V : membrane potential [mV].
        u : recovery variable [mV].
        sp : spike state. 
        sp_t : last spike time.
        inp : input, including external and synaptic inputs.
        '''
    )

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

    @bp.integrate
    def int_u(u, t, V):
        return a * (b * V - u)

    @bp.integrate
    def int_V(V, t, u, Isyn):
        dfdt = 0.04 * V * V + 5 * V + 140 - u + Isyn
        dgdt = noise
        return dfdt, dgdt

    if np.any(ref > 0.):

        def update(ST, _t_):
            if (_t_ - ST['sp_t']) > ref:
                V = int_V(ST['V'], _t_, ST['u'], ST['inp'])
                u = int_u(ST['u'], _t_, ST['V'])
                if V >= Vth:
                    V = c
                    u += d
                    ST['sp_t'] = _t_
                    ST['sp'] = True
                ST['V'] = V
                ST['u'] = u
                ST['inp'] = 0.

    else:

        def update(ST, _t_):
            V = int_V(ST['V'], _t_, ST['u'], ST['inp'])
            u = int_u(ST['u'], _t_, ST['V'])
            if V >= Vth:
                V = c
                u += d
                ST['sp_t'] = _t_
                ST['sp'] = True
            ST['V'] = V
            ST['u'] = u
            ST['inp'] = 0.

    return bp.NeuType(name='Izhikevich', requires={'ST': state}, steps=update, vector_based=False)


if __name__ == '__main__':
    bp.profile.set(backend='numba', merge_steps=True)

    Izhikevich = define_Izhikevich()
    neu = bp.NeuGroup(Izhikevich, 10, monitors=['V', 'u'])
    neu.pars['noise'] = 1.

    neu.run(duration=100, inputs=['inp', 20], report=True)

    indexes = [0, 1, 2]
    fig, gs = bp.visualize.get_figure(2, 1, 3, 12)

    fig.add_subplot(gs[0, 0])
    bp.visualize.plot_potential(neu.mon, neu.mon.ts, neuron_index=indexes)
    plt.xlim(-0.1, 100.1)
    plt.legend()

    fig.add_subplot(gs[1, 0])
    bp.visualize.plot_value(neu.mon, neu.mon.ts, 'u', val_index=indexes)
    plt.xlim(-0.1, 10.1)
    plt.xlabel('Time (ms)')
    plt.legend()

    plt.show()
