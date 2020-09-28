import matplotlib.pyplot as plt

import npbrain as nn

npbrain.profile.set_backend('numba')
npbrain.profile.ftype = 'float32'


def run_ampa(cls, num_pre=5, num_post=10, prob=1., duration=650.):
    pre = nn.FreqInput(num_pre, 10., 100.)
    post = nn.generate_fake_neuron(num_post)
    conn = nn.connect.fixed_prob(pre.num, post.num, prob)
    ampa = cls(pre, post, conn, delay=10.)
    mon = nn.StateMonitor(ampa, ['g_out', 's', 'g_in'])
    net = nn.Network(ampa, pre, post, mon)

    net.run(duration, report=True)

    ts = net.ts()
    fig, gs = nn.visualize.get_figure(1, 1, 5, 10)
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, mon.g_out[:, 0], lw=3, label='g_out')
    plt.plot(ts, mon.g_in[:, 0], lw=1, label='g_in')
    plt.plot(ts, mon.s[:, 0], label='s')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    for num in [1, 10]:
        run_ampa(nn.AMPA1, num, num)
        run_ampa(nn.AMPA2, num, num)
