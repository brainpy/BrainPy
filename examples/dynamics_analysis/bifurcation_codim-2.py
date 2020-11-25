# -*- coding: utf-8 -*-

from collections import OrderedDict

import matplotlib.pyplot as plt

import brainpy as nb
from brainpy.dynamics import BifucationAnalyzer

nb.profile.set_dt(0.02)
nb.profile.merge_integral = False


def dummy_co2_model(noise=0.):
    r'''
    A dummy 1D test neuronal model for codimension 2 bifurcation testing.

    .. math::
    \dot{x} = \mu+\lambda x - x**3
    '''

    lambda_ = 0
    mu = 0

    ST = nb.types.NeuState({'x': 0.})

    @nb.integrate
    def int_x(x, t):
        dxdt = mu + lambda_ * x - x ** 3
        return dxdt

    def update(ST, _t_):
        x = int_x(ST['x'], _t_)
        ST['x'] = x

    return nb.NeuType(name="dummy_co2_model", requires=dict(ST=ST), steps=update, vector_based=True)


param_dict = OrderedDict()
param_dict["mu"] = [-4, 4]
param_dict["lambda_"] = [-1, 4]

res_dict = OrderedDict()
res_dict["mu"] = 50
res_dict["lambda_"] = 50

an = BifucationAnalyzer(neuron=dummy_co2_model(),
                        var_lim={"x": [-3, 3]},
                        plot_var="x",
                        parameter=param_dict,
                        resolution=res_dict)
an.run()
plt.show()
