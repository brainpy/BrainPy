import matplotlib.pyplot as plt

import npbrain as nn

npbrain.profile.set_backend('numba')


def run_nmda(cls, num_pre=5, num_post=10, prob=1., monitor=[],
             run_duration=300, stimulus_gap=10):
    pre = nn.FreqInput(num_pre, freq=1e3 / stimulus_gap, start_time=20.)
    post = nn.generate_fake_neuron(num_post)
    conn = nn.connect.fixed_prob(num_pre, num_post, prob)
    nmda = cls(pre, post, conn, delay=2.)
    mon = nn.StateMonitor(nmda, monitor)
    net = nn.Network(nmda, pre, post, mon)
    net.run(run_duration, report=True)

    fig, gs = nn.visualize.get_figure(1, 1, 5, 10)
    fig.add_subplot(gs[0, 0])
    for k in monitor:
        plt.plot(net.ts(), getattr(mon, k)[:, 0], label=k)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    for num in [1, 10]:
        run_nmda(nn.NMDA, num_pre=num, num_post=num,
                 stimulus_gap=30, monitor=['g_out', 'x', 's'])
