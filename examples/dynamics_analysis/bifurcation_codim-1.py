# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import brainpy as nb
from brainpy.dynamics import BifucationAnalyzer

nb.profile.set_dt(0.02)
nb.profile.merge_integral = False


def define_dummy1d_model(noise=0.):
    r'''
    A dummy 1D test neuronal model.

    .. math::
    \dot{x} = x**3-x
    '''

    ST = nb.types.NeuState(
        {'x': -10, 'inp': 0.}
    )

    @nb.integrate
    def int_x(x, t, inp):
        dxdt = x ** 3 - x + inp
        return dxdt

    def update(ST, _t_):
        x = int_x(ST['x'], _t_, ST['inp'])
        ST['inp'] = 0.

    return nb.NeuType(name="dummy_model", requires=dict(ST=ST), steps=update, vector_based=True)


an = BifucationAnalyzer(neuron=define_dummy1d_model(),
                        var_lim={"x": [-2, 2]},
                        plot_var="x",
                        parameter={"inp": [-0.5, 0.5]},
                        resolution={"inp": 500})

an.run()
plt.show()
