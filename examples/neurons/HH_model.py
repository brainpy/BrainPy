
import npbrain as nn
import npbrain._numpy as np

nn.profile.set_backend('numba')
nn.profile.set_dt(0.02)

import matplotlib.pyplot as plt


def HH(method=None, noise=0., E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387,
       g_Leak=0.03, C=1.0, Vr=-65., Vth=20.):
    """The Hodgkin–Huxley neuron model.

    The Hodgkin–Huxley model can be thought of as a differential equation
    with four state variables, :math:`v(t)`, :math:`m(t)`, :math:`n(t)`, and
    :math:`h(t)`, that change with respect to time :math:`t`.

    Parameters
    ----------
    method : str, callable, dict
        The numerical integration method. Either a string with the name of a
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

    var2index = {'V': Vr, 'm': 0., 'h': 0., 'n': 0., 'sp': False, 'pre_above_th': False}

    @nn.core.integrate(method=method)
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m

    @nn.core.integrate(method=method)
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h

    @nn.core.integrate(method=method)
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n

    @nn.core.integrate(method=method, noise=noise / C)
    def int_V(V, t, m, h, n, Isyn):
        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + Isyn) / C
        return dvdt

    def update_state(neu_state, t):
        V, Isyn = neu_state[0], neu_state[-1]
        m = np.clip(int_m(neu_state[1], t, V), 0., 1.)
        h = np.clip(int_h(neu_state[2], t, V), 0., 1.)
        n = np.clip(int_n(neu_state[3], t, V), 0., 1.)
        V = int_V(V, t, m, h, n, Isyn)
        neu_state[0] = V
        neu_state[1] = m
        neu_state[2] = h
        neu_state[3] = n
        judge_spike(neu_state, Vth, t)

    return Neurons(**locals())



if __name__ == '__main__':
    hh = nn.HH(1, noise=1.)
    mon = nn.StateMonitor(hh, ['V', 'm', 'h', 'n'])
    net = nn.Network(hh=hh, mon=mon)
    net.run(duration=100, inputs=[hh, -10], report=True)

    ts = net.ts()
    fig, gs = nn.visualize.get_figure(2, 1, 3, 12)

    fig.add_subplot(gs[0, 0])
    plt.plot(ts, mon.V[:, 0], label='N')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(ts, mon.m[:, 0], label='m')
    plt.plot(ts, mon.h[:, 0], label='h')
    plt.plot(ts, mon.n[:, 0], label='n')
    plt.legend()
    plt.xlim(-0.1, net._run_time + 0.1)
    plt.xlabel('Time (ms)')

    plt.show()

