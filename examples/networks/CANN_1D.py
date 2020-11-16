# -*- coding: utf-8 -*-

"""
Implementation of the paper:

- Si Wu, Kosuke Hamaguchi, and Shun-ichi Amari. "Dynamics and computation
  of continuous attractors." Neural computation 20.4 (2008): 994-1025.

The mathematical equation of the Continuous-attractor Neural Network (CANN) is
given by:

\tau \frac{du(x,t)}{dt} = -u(x,t) + \rho \int dx^\prime J\left(x,x^\prime\right)
r\left(x^\prime,t\right)+A\exp\left[-\frac{\left|x-z(t)\right|^2}{4a^2}\right]\\

r(x,t) = \frac{u(x,t)^2}{1 + k \rho \int dx^\prime u(x^\prime,t)^2}\\

J\left(x,x^\prime\right) = \frac{1}{\sqrt{2\pi}a}\exp\left(
-\frac{\left|x-x^\prime\right|^2}{2a^2}\right)

"""

import brainpy as bp
import brainpy.numpy as np

tau = 1.
k = 0.5  # Degree of the rescaled inhibition
a = 0.5  # Half-width of the range of excitatory connections
A = 0.5  # Magnitude of the rescaled external input
z_min = -np.pi
z_max = np.pi
z_range = z_max - z_min
rho = 2 / (np.sqrt(2 * np.pi) * a)


def dist(d):
    d = np.remainder(d, z_range)
    d = np.where(d > 0.5 * z_range, d - z_range, d)
    return d


# neuron #
# ------ #


@bp.integrate
def int_u(u, t, Jxx, Iext):
    r1 = np.square(u)
    r2 = 1.0 + k * rho * np.sum(r1) * dx
    r = r1 / r2
    Irec = rho * np.dot(Jxx, r) * dx
    dudt = (-u + Irec + Iext) / tau
    return (dudt,), r


def neu_update(ST, _t_, Jxx):
    ST['u'], ST['r'] = int_u(ST['u'], _t_, Jxx, ST['input'])
    ST['input'] = 0.


requires = {
    'ST': bp.types.NeuState(['x', 'u', 'r', 'input']),
    'Jxx': bp.types.Array(dim=2, help='Weight connection matrix.')
}

cann = bp.NeuType(name='CANN',
                  steps=neu_update,
                  requires=requires,
                  vector_based=True)


# connection #
# ---------- #


def make_conn(x):
    assert np.ndim(x) == 1
    x_left = np.reshape(x, (len(x), 1))
    x_right = np.tile(x, (len(x), 1))
    d = dist(x_left - x_right)
    jxx = np.exp(-0.5 * np.square(d / a)) / (np.sqrt(2 * np.pi) * a)
    return jxx


# network #
# ------- #

def population_encoding():
    group = bp.NeuGroup(cann, geometry=512, monitors=['r'])
    group.ST['x'] = np.linspace(z_min, z_max, group.num)
    group.Jxx = make_conn(group.ST['x'])
    global dx
    dx = z_range / group.num

    I1 = A * np.exp(-0.25 * np.square(dist(group.ST['x'] - 0.) / a))
    Iext, duration = bp.inputs.constant_current([(0., 10.), (I1, 50.), (0., 10.)])
    group.run(duration=duration, inputs=('ST.input', Iext))

    bp.visualize.animate_1D(dynamical_vars=[{'ys': group.mon.r, 'xs': group.ST['x'], 'legend': 'r'},
                                            {'ys': Iext, 'xs': group.ST['x'], 'legend': 'Iext'}],
                            show=True,
                            frame_step=5,
                            frame_delay=50, )


def population_decoding():
    group = bp.NeuGroup(cann, geometry=512, monitors=['r'])
    group.ST['x'] = np.linspace(z_min, z_max, group.num)
    group.Jxx = make_conn(group.ST['x'])
    global dx
    dx = z_range / group.num

    I1 = A * np.exp(-0.25 * np.square(dist(group.ST['x'] - 0.) / a)) + 0.005 * np.random.randn(group.num)
    group.run(duration=100., inputs=('ST.input', I1))

    bp.visualize.animate_1D(dynamical_vars={'ys': group.mon.r, 'xs': group.ST['x'], 'legend': 'r'},
                            static_vars={'ys': I1, 'xs': group.ST['x'], 'legend': 'Iext'},
                            show=True,
                            frame_step=5,
                            frame_delay=50, )


def smooth_tracking():
    group = bp.NeuGroup(cann, geometry=512, monitors=['r'])
    group.ST['x'] = np.linspace(z_min, z_max, group.num)
    group.Jxx = make_conn(group.ST['x'])
    global dx
    dx = z_range / group.num

    duration = 150.
    position = bp.inputs.ramp_current(c_start=0, c_end=30., duration=duration, t_start=20., t_end=100.)
    position = np.reshape(dist(position), (-1, 1))
    I1 = A * np.exp(-0.25 * np.square(dist(group.ST['x'] - position) / a))
    group.run(duration=duration, inputs=('ST.input', I1))

    bp.visualize.animate_1D(dynamical_vars=[{'ys': group.mon.r, 'xs': group.ST['x'], 'legend': 'r'},
                                            {'ys': I1, 'xs': group.ST['x'], 'legend': 'Iext'}],
                            show=True,
                            frame_step=5,
                            frame_delay=50, )


if __name__ == '__main__':
    pass
    # population_encoding()
    # population_decoding()
    smooth_tracking()
