import numpy as np
import matplotlib.pyplot as plt

import npbrain as nn

nn.profile.set_backend('numpy')


def run_stp(cls, num_pre=5, num_post=10, weights=5.,
            prob=1., monitor=[], run_duration=300, stimulus_gap=10):
    pre = nn.generate_fake_neuron(num_pre)
    post = nn.generate_fake_neuron(num_post)
    stp = cls(pre, post, weights, {'method': 'fixed_prob', 'prob': prob},
              delay=2.)
    mon = nn.StateMonitor(stp, monitor)

    run_duration, dt = run_duration, 0.1
    stimulus_gap = int(stimulus_gap / dt)
    ts = np.arange(0, run_duration, dt)
    mon.init_state(len(ts))
    for i, t in enumerate(ts):
        stp.state[0][-1, :] = 0.
        if (i + 1) % stimulus_gap == 0:
            stp.state[0][-1, :] = 1.
        stp.update_state(stp.state, stp.var2index_array, t)
        mon.update_state(stp.state, mon.state, mon.target_index_by_vars(), i)

    fig, gs = nn.visualize.get_figure(1, 1, 5, 10)
    fig.add_subplot(gs[0, 0])
    for k in monitor:
        plt.plot(ts, getattr(mon, k)[:, 0], label=k)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    for num in [1, 10]:
        run_stp(nn.STP, num_pre=num, num_post=num, stimulus_gap=50,
                monitor=['g_post', 'x', 'u'])
