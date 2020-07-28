import matplotlib.pyplot as plt

import npbrain as nn

nn.profile.set_backend('numba')


def run_gaba(cls, num_pre=5, num_post=10, prob=1., monitor=[],
             duration=300, stimulus_gap=10):
    pre = nn.FreqInput(num_pre, 1e3 / stimulus_gap, 20.)
    post = nn.generate_fake_neuron(num_post)
    conn = nn.connect.fixed_prob(num_pre, num_post, prob)
    gaba = cls(pre, post, conn, delay=2.)
    mon = nn.StateMonitor(gaba, monitor)

    net = nn.Network(gaba, pre, post, mon)

    net.run(duration, report=True)

    fig, gs = nn.visualize.get_figure(1, 1, 5, 10)
    fig.add_subplot(gs[0, 0])
    for k in monitor:
        plt.plot(net.run_time(), getattr(mon, k)[:, 0], label=k)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    for num in [1, 10]:
        run_gaba(nn.GABAa1, num_pre=num, num_post=num, stimulus_gap=30,
                 monitor=['g_out', 's'])

        run_gaba(nn.GABAa2, num_pre=num, num_post=num, stimulus_gap=30,
                 monitor=['g_out', 's'])

        run_gaba(nn.GABAb1, num_pre=num, num_post=num,
                 duration=2000, stimulus_gap=50,
                 monitor=['g_out', 'R', 'G'])

        run_gaba(nn.GABAb2, num_pre=num, num_post=num,
                 duration=2000, stimulus_gap=50,
                 monitor=['g_out', 'D', 'R', 'G'])
