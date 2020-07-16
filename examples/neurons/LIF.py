import npbrain as nn

import numpy as np
import matplotlib.pyplot as plt

nn.profile.set_backend('numba')
nn.profile.set_dt(0.02)
np.random.seed(1234)

if __name__ == '__main__':
    lif1 = nn.LIF(500, ref=1., noise=1.1)
    lif2 = nn.LIF(1000, ref=1., noise=1.1)
    conn = nn.conn.fixed_prob(lif1.num, lif2.num, prob=0.1)
    syn = nn.VoltageJumpSynapse(lif1, lif2, 0.2, conn)
    mon_lif1 = nn.StateMonitor(lif1)
    mon2 = nn.SpikeMonitor(lif1)
    mon_lif2 = nn.StateMonitor(lif2)
    mon4 = nn.SpikeMonitor(lif2)
    net = nn.Network(syn=syn, lif1=lif1, lif2=lif2, mon1=mon_lif1, mon2=mon2,
                     mon3=mon_lif2, mon4=mon4)
    net.run(duration=100, inputs=[lif1, 15], report=True)

    ts = net.run_time()
    fig, gs = nn.vis.get_figure(2, 1, 3, 8)

    ax = fig.add_subplot(gs[0, 0])
    nn.vis.plot_potential(mon_lif1, ts, (0, 1), ax)
    plt.xlim(-0.1, net.current_time + 0.1)
    plt.title('LIF neuron group 1')
    plt.legend()

    ax = fig.add_subplot(gs[1, 0])
    if len(mon2.time):
        nn.vis.plot_raster(mon2, ax=ax)
    plt.xlim(-0.1, net.current_time + 0.1)

    plt.show()

if __name__ == '__main__':
    npbrain.utils.profile.set_dt(0.02)
    lif1 = nn.LIF(500, ref=1., noise=1.1)
    lif2 = nn.LIF(1000, ref=1., noise=1.1)
    conn = nn.conn.fixed_prob(lif1.num, lif2.num, prob=0.1)
    syn = nn.VoltageJumpSynapse(lif1, lif2, 0.2, conn)
    mon_lif1 = nn.StateMonitor(lif1, ['V', 'spike'])
    mon_lif2 = nn.StateMonitor(lif2, ['V', 'spike'])
    net = nn.Network(syn=syn, lif1=lif1, lif2=lif2, mon1=mon_lif1, mon2=mon_lif2)
    net.run(duration=100, inputs=[lif1, 15], report=True)

    ts = net.run_time()
    fig, gs = nn.vis.get_figure(2, 2, 3, 6)

    ax1 = fig.add_subplot(gs[0, 0])
    nn.vis.plot_potential(mon_lif1, ts, (0, 1), ax1)
    plt.xlim(-0.1, net.current_time + 0.1)
    plt.title('LIF neuron group 1')
    plt.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    nn.vis.plot_potential(mon_lif2, ts, (0, 1), ax2)
    plt.xlim(-0.1, net.current_time + 0.1)
    plt.title('LIF neuron group 2')
    plt.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    nn.vis.plot_raster(mon_lif1, ts, ax=ax3)
    plt.xlim(-0.1, net.current_time + 0.1)

    ax4 = fig.add_subplot(gs[1, 1])
    nn.vis.plot_raster(mon_lif2, ts, ax=ax4)
    plt.xlim(-0.1, net.current_time + 0.1)
    plt.show()

if __name__ == '__main__':
    npbrain.utils.profile.set_dt(0.02)
    lif1 = nn.LIF(500, ref=1., noise=1.1)
    mon = nn.SpikeMonitor(lif1)
    net = nn.Network(lif1=lif1, mon=mon)
    inputs = np.linspace(10.5, 40., lif1.num)
    net.run(duration=100, inputs=[lif1, inputs], report=True)

    fig, gs = nn.vis.get_figure(1, 1, 4, 8)
    fig.add_subplot(gs[0, 0])
    plt.plot(mon.time, mon.index, '|', markersize=4)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.show()
