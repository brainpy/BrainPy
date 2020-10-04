import matplotlib.pyplot as plt

import npbrain as npb
import npbrain._numpy as np

npb.profile.set_backend('numpy')
npb.profile.set_dt(0.02)
npb.profile.show_codgen = True


def define(method=None, noise=0., E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387,
           g_Leak=0.03, C=1.0, Vr=-65., Vth=20.):
    """The Hodgkin–Huxley neuron model.

    The Hodgkin–Huxley model can be thought of as a differential equation
    with four state variables, :math:`v(t)`, :math:`m(t)`, :math:`n(t)`, and
    :math:`h(t)`, that change with respect to time :math:`t`.

    Parameters
    ----------
    method : str, callable, dict
        The numerical integrator method. Either a string with the name of a
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

    Returns
    -------
    return_dict : dict
        The necessary variables.
    """

    attrs = dict(
        ST=npb.types.NeuState({'V': Vr, 'm': 0., 'h': 0., 'n': 0., 'sp': 0., 'inp': 0.},
                              help='Hodgkin–Huxley neuron state.'),
    )

    @npb.integrator.integrate(method=method)
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m

    @npb.integrator.integrate(method=method)
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h

    @npb.integrator.integrate(method=method)
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n

    @npb.integrator.integrate(method=method, noise=noise / C)
    def int_V(V, t, m, h, n, Isyn):
        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + Isyn) / C
        return dvdt

    def update(ST, t):
        m = np.clip(int_m(ST['m'], t, ST['V']), 0., 1.)
        h = np.clip(int_h(ST['h'], t, ST['V']), 0., 1.)
        n = np.clip(int_n(ST['n'], t, ST['V']), 0., 1.)
        V = int_V(ST['V'], t, m, h, n, ST['inp'])
        sp = np.logical_and(ST['V'] < Vth, V >= Vth)
        ST['sp'] = sp
        ST['V'] = V
        ST['m'] = m
        ST['h'] = h
        ST['n'] = n
        ST['inp'] = 0.

    return {'attrs': attrs, 'step_func': update}


HH = npb.NeuType(name='HH_neuron', create_func=define, group_based=True)

import inspect
a = define()
vars = inspect.getclosurevars(a['step_func'])
print()



if __name__ == '__main__1':
    neu = npb.NeuGroup(HH, geometry=(1, ), monitors=['sp', 'V', 'm', 'h', 'n'])
    net = npb.Network(neu)
    net.run(duration=100., inputs=[neu, 'ST.inp', 10.], report=True)

    ts = net.ts
    fig, gs = npb.visualize.get_figure(2, 1, 3, 12)

    fig.add_subplot(gs[0, 0])
    plt.plot(ts, neu.mon.V[:, 0], label='N')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(ts, neu.mon.m[:, 0], label='m')
    plt.plot(ts, neu.mon.h[:, 0], label='h')
    plt.plot(ts, neu.mon.n[:, 0], label='n')
    plt.legend()
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')

    plt.show()
