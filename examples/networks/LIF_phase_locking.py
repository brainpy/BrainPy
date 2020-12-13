# -*- coding: utf-8 -*-

"""
Implementation of the paper:

- Brette, Romain. "Dynamics of one-dimensional spiking neuron
  models." Journal of mathematical biology 48.1 (2004): 38-56.

"""

import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.numpy as np

bp.profile.set(backend='numba')

tau = 100.  # ms
Vth = 1.  # mV
Vr = 0.  # mV


@bp.integrate
def int_f(V, t, Isyn):
    return (-V + Isyn + 2 * np.sin(2 * np.pi * t / tau)) / tau


def update(ST, _t_):
    V = int_f(ST['V'], _t_, ST['input'])
    spike = V >= Vth
    ST['spike'] = spike
    ST['V'] = np.where(spike, Vr, V)
    ST['input'] = 0.


ST = bp.types.NeuState({'V': 0, 'spike': 0., 'input': 0.})
lif = bp.NeuType(name='LIF',
                 requires=dict(ST=ST),
                 steps=update)

group = bp.NeuGroup(lif, geometry=2000, monitors=['spike'])
inputs = np.linspace(2., 4., group.num)


def method1():
    group.run(duration=10 * 1000., inputs=('input', inputs), report=True)
    idx = int(5 * 1000. / bp.profile.get_dt())
    return group.mon.spike[idx:], group.mon.ts[idx:]


def method2():
    net = bp.Network(group, mode='repeat')
    net.run(duration=5 * 1000., inputs=(group, 'input', inputs), report=True)
    net.run(duration=5 * 1000., inputs=(group, 'input', inputs), report=True)
    return group.mon.spike, group.mon.ts


def show(spikes, times):
    indices, times = bp.measure.raster_plot(spikes, times)
    plt.plot((times % tau) / tau, inputs[indices], ',')
    plt.xlabel('Spike phase')
    plt.ylabel('Parameter (input)')
    plt.show()


# show(*method1())
show(*method2())


