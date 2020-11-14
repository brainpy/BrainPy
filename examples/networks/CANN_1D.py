# -*- coding: utf-8 -*-

"""
Implementation of the paper:

- Wu, Si, Kosuke Hamaguchi, and Shun-ichi Amari. "Dynamics and computation
  of continuous attractors." Neural computation 20.4 (2008): 994-1025.

The mathematical equation of the Continuous-attractor Neural Network (CANN) is
given by:

\tau \frac{du(x,t)}{dt} = -u(x,t) + \int dx^\prime J\left(x,x^\prime\right)
r\left(x^\prime,t\right)+A\exp\left[-\frac{\left|x-z(t)\right|^2}{4a^2}\right]

r(x,t) = \frac{\left[u(x,t)\right]+{}^2}{1+\frac{k}{8\sqrt{2\pi}a}\int dx^\prime
\left[u(x^\prime,t)\right]+{}^2}

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


def dist(d):
    d = np.remainder(d, z_range)
    d = np.where(d > 0.5 * z_range, d - z_range, d)
    return d


# neuron #
# ------ #


@bp.integrate
def int_u(u, t, Jxx, Iext):
    r = np.square(0.5 * (u + np.abs(u)))
    B = 1.0 + 0.125 * k * np.sum(r) * dx / (np.sqrt(2 * np.pi) * a)
    Irec = np.dot(Jxx, r / B) * dx
    dudt = (-u + Irec + Iext) / tau
    return dudt


def neu_update(ST, _t_, Jxx):
    ST['u'] = int_u(ST['u'], _t_, Jxx, ST['input'])
    ST['input'] = 0.


cann = bp.NeuType(name='CANN',
                  steps=neu_update,
                  requires=dict(ST=bp.types.NeuState(['u', 'x', 'input'])),
                  vector_based=True)


# connection #
# ---------- #


def make_conn(num, x):
    conn = np.zeros((num, num))
    for i in range(num):
        d = dist(x[i] - x)
        jxx = np.exp(-0.5 * np.square(d / a)) / (np.sqrt(2 * np.pi) * a)
        conn[i] = jxx
    return conn


# network #
# ------- #

group = bp.NeuGroup(cann, geometry=128, monitors=['u'])
group.ST['x'] = np.linspace(z_min, z_max, group.num)
group.ST['u'] = np.sqrt(32.0) * np.exp(-0.25 * np.square(dist(group.ST['x'] - 0.) / a))
group.Jxx = make_conn(group.num, group.ST['x'])
dx = z_range / group.num

I1 = A * np.exp(-0.25 * np.square(dist(group.ST['x'] - 0.) / a))
I2 = A * np.exp(-0.25 * np.square(dist(group.ST['x'] - 0.5 * np.pi) / a))
Iext, duration = bp.inputs.constant_current([(I1, 200.), (I2, 200)])
group.run(duration=duration, inputs=('ST.input', Iext))

bp.visualize.animate_1D(group.mon.u, xticks=group.ST['x'], show=True, frame_step=10)
