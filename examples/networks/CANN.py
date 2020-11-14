# -*- coding: utf-8 -*-

"""
Implementation of "Wu, Si, Kosuke Hamaguchi, and Shun-ichi Amari.
"Dynamics and computation of continuous attractors." Neural
computation 20.4 (2008): 994-1025."
"""

import brainpy as bp
import brainpy.numpy as np

tau = 1.
k = 0.5
dx = 0.1
a = 0.5
A = 0.5


@bp.integrate
def int_u(u, t, Isyn):
    return (-u + Isyn) / tau


def neu_update(ST, _t_):
    u = int_u(ST['u'], _t_, ST['input'])
    r = np.square(0.5 * (u + np.abs(u)))
    B = 1.0 + 0.125 * k * np.sum(r) * dx / (np.sqrt(2 * np.pi) * a)
    ST['r'] = r / B
    ST['u'] = u
    ST['input'] = 0.


cann_neuron = bp.NeuType(name='CANN_neuron',
                         steps=neu_update,
                         requires=dict(ST=bp.types.NeuState(['u', 'r', 'x', 'input'])),
                         vector_based=True)


def syn_update(ST, pre):
    g = np.dot(ST['J'], pre['r']) * dx
    ST['g'] = g
    pre['input'] += g


cann_synapse = bp.SynType(name='CANN_synapse',
                          steps=syn_update,
                          requires=dict(ST=bp.types.SynState(['J', 'g'])),
                          vector_based=True)

group = bp.NeuGroup(cann_neuron, geometry=128, monitors=['u'])
conn = bp.SynConn(cann_synapse, pre_group=group, post_group=group,
                  conn=bp.connect.GaussianWeight(sigma=a, w_max=A))

CANN = bp.Network(group, conn)
CANN.run(duration=1000., inputs=())
