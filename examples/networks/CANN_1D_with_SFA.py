# -*- coding: utf-8 -*-

"""
Implementation of the paper:

- Si Wu, Kosuke Hamaguchi, and Shun-ichi Amari. "Dynamics and computation
  of continuous attractors." Neural computation 20.4 (2008): 994-1025.

The mathematical equation of the Continuous-attractor Neural Network (CANN) is
given by:

\tau \frac{du(x,t)}{dt} = -u(x,t) + \rho \int dx^\prime J\left(x,x^\prime\right)
r\left(x^\prime,t\right) - V(x,t) +A\exp\left[-\frac{\left|x-z(t)\right|^2}{4a^2}\right]\\

\tau_{v} \frac{d V(x, t)}{d t}=-V(x, t)+m U(x, t) \\

r(x,t) = \frac{u(x,t)^2}{1 + k \rho \int dx^\prime
u(x^\prime,t)^2}\\

J\left(x,x^\prime\right) = \frac{1}{\sqrt{2\pi}a}\exp\left(
-\frac{\left|x-x^\prime\right|^2}{2a^2}\right)

"""

import brainpy as bp
import numpy as np

tau = 1.
tau_v = 48.
k = 0.5  # Degree of the rescaled inhibition
a = 0.5  # Half-width of the range of excitatory connections
A = 0.5  # Magnitude of the rescaled external input
z_min = -np.pi
z_max = np.pi
z_range = z_max - z_min
rho = 2 / (np.sqrt(2 * np.pi) * a)
m = tau / tau_v
dx = 0.


def dist(d):
    d = np.remainder(d, z_range)
    d = np.where(d > 0.5 * z_range, d - z_range, d)
    return d


# neuron #
# ------ #


@bp.integrate
def int_u(u, t, v, Jxx, Iext):
    r1 = np.square(u)
    r2 = 1.0 + k * rho * np.sum(r1) * dx
    r = r1 / r2
    Irec = rho * np.dot(Jxx, r) * dx
    dudt = (-u + Irec - v + Iext) / tau
    return (dudt,), r


@bp.integrate
def int_v(v, t, u):
    return (-v + m * u) / tau_v


def neu_update(ST, _t_, Jxx):
    ST['u'], ST['r'] = int_u(ST['u'], _t_, ST['v'], Jxx, ST['input'])
    ST['v'] = int_v(ST['v'], _t_,ST['u'])
    ST['input'] = 0.


requires = {
    'ST': bp.types.NeuState(['x', 'v', 'u', 'r', 'input']),
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
