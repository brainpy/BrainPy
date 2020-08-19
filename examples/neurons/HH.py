import npbrain as nn

nn.profile.set_backend('numba')
nn.profile.set_dt(0.02)

import matplotlib.pyplot as plt

if __name__ == '__main__':
    hh = nn.HH(1, noise=1.)
    mon = nn.StateMonitor(hh, ['V', 'm', 'h', 'n'])
    net = nn.Network(hh=hh, mon=mon)
    net.run(duration=100, inputs=[hh, -10], report=True)

    ts = net.run_time()
    fig, gs = nn.visualize.get_figure(2, 1, 3, 12)

    fig.add_subplot(gs[0, 0])
    plt.plot(ts, mon.V[:, 0], label='N')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net.current_time + 0.1)
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(ts, mon.m[:, 0], label='m')
    plt.plot(ts, mon.h[:, 0], label='h')
    plt.plot(ts, mon.n[:, 0], label='n')
    plt.legend()
    plt.xlim(-0.1, net.current_time + 0.1)
    plt.xlabel('Time (ms)')

    plt.show()

