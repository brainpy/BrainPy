# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    'constant_current',
    'pulse_current',
    'ramp_current',
    'format_current',
]


def constant_current(Iext, dt):
    total_duration = sum([a[1] for a in Iext])
    current = np.zeros(int(np.ceil(total_duration / dt)))
    start = 0
    for c_size, duration in Iext:
        length = int(duration / dt)
        current[start: start + length] = c_size
        start += length
    return current, total_duration



def pulse_current(p_points, p_duration, p_size, duration, dt):
    current = np.zeros(int(np.ceil(duration / dt)))
    p_len = int(p_duration / dt)
    for p in p_points:
        pp = int(p / dt)
        current[pp: pp + p_len] = p_size
    return current



def ramp_current(c_start, c_end, duration, dt, t_start=0, t_end=None):
    t_end = duration if t_end is None else t_end
    current = np.zeros(int(np.ceil(duration / dt)))
    p1 = int(np.ceil(t_start / dt))
    p2 = int(np.ceil(t_end / dt))
    current[p1: p2] = np.linspace(c_start, c_end, p2 - p1)
    return current


def format_current(Iext, dt):
    if len(Iext) == 1 or isinstance(Iext[0], type(Iext[1])):
        current, duration = step_current(Iext, dt)
    elif len(Iext) == 3:
        c_start, c_end, duration = Iext
        current = ramp_current(c_start, c_end, duration, dt)
    elif len(Iext) == 4:
        p_points, p_duration, p_size, duration = Iext
        current = shock_current(p_points, p_duration, p_size, duration, dt)
    elif len(Iext) == 5:
        c_start, c_end, duration, t_start, t_end = Iext
        current = ramp_current(c_start, c_end, duration, dt, t_start, t_end)
    else:
        raise ValueError('Unknown "Iext": {}'.format(Iext))
    return current, duration


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    sys.path.append('../../')
    from npbrain.utils.vis import get_figure

# test `step_current`
if __name__ == '__main__1':
    fig, gs = get_figure(2, 1)

    current, duration = step_current([(0, 100), (1, 300), (0, 100)], 0.1)
    ts = np.arange(0, duration, 0.1)
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, current)
    plt.title('[(0, 100), (1, 300), (0, 100)]')

    current, duration = step_current([(-1, 10), (1, 3), (3, 30), (-0.5, 10)], 0.1)
    ts = np.arange(0, duration, 0.1)
    fig.add_subplot(gs[1, 0])
    plt.plot(ts, current)
    plt.title('[(-1, 10), (1, 3), (3, 30), (-0.5, 10)]')

    plt.show()

# test `ramp_current`
if __name__ == '__main__1':
    fig, gs = get_figure(2, 1)

    duration, dt = 1000, 0.1
    current = ramp_current(0, 1, duration, dt)
    ts = np.arange(0, duration, dt)
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, current)
    plt.title(r'$c_{start}$=0, $c_{end}$=%d, duration, dt=%.1f, '
              r'$t_{start}$=0, $t_{end}$=None' % (duration, dt,))

    duration, dt, t_start, t_end = 1000, 0.1, 200, 800
    current = ramp_current(0, 1, duration, dt, t_start, t_end)
    ts = np.arange(0, duration, dt)
    fig.add_subplot(gs[1, 0])
    plt.plot(ts, current)
    plt.title(r'$c_{start}$=0, $c_{end}$=1, duration=%d, dt=%.1f, '
              r'$t_{start}$=%d, $t_{end}$=%d' % (duration, dt, t_start, t_end))

    plt.show()

# test `shock_current`
if __name__ == '__main__1':
    fig, gs = get_figure(1, 1)

    points, size, duration, dt = [10, 20, 30, 200, 300], 0.5, 1000, 0.1
    current = shock_current(points, 1, size, duration, dt)
    ts = np.arange(0, duration, dt)
    fig.add_subplot(gs[0, 0])
    plt.plot(ts, current)
    plt.title(r'points=%s, duration=%d' % (points, duration))

    plt.show()
