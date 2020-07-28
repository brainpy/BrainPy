import numpy as np
import matplotlib.pyplot as plt

import npbrain as nn

nn.profile.set_backend('numpy')


def run_stp(cls, num_pre=5, num_post=10, weights=5.,
            prob=1., monitor=[], run_duration=300, stimulus_gap=10):
    pre = nn.FreqInput(num_pre, freq=1e3 / stimulus_gap, start_time=20.)
    post = nn.generate_fake_neuron(num_post)
    conn = nn.connect.fixed_prob(pre.num, post.num, prob=prob)
    stp = nn.STP(pre, post, weights, connection=conn, delay=2.)
    mon = nn.StateMonitor(stp, monitor)

    net = nn.Network(stp, pre, post, mon)
    net.run(run_duration)

    fig, gs = nn.visualize.get_figure(1, 1, 5, 10)
    fig.add_subplot(gs[0, 0])
    for k in monitor:
        plt.plot(net.run_time(), getattr(mon, k)[:, 0], label=k)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    for num in [1, 10]:
        run_stp(nn.STP, num_pre=num, num_post=num, stimulus_gap=50,
                monitor=['g_post', 'x', 'u'])
