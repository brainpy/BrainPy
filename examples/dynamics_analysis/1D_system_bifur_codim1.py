# -*- coding: utf-8 -*-

import brainpy as bp


def define_model():
    """
    A dummy 1D test neuronal model.

    .. math::

        \dot{x} = x**3-x + I

    """

    @bp.integrate
    def int_x(x, t, input):
        dxdt = x ** 3 - x + input
        return dxdt

    def update(ST, _t):
        ST['x'] = int_x(ST['x'], _t, ST['input'])
        ST['input'] = 0.

    return bp.NeuType(name="dummy_model",
                      ST=bp.types.NeuState({'x': -10, 'input': 0.}),
                      steps=update)


an = bp.BifurcationAnalyzer(
    model=define_model(),
    target_pars={'input': [-0.5, 0.5]},
    dynamical_vars={"x": [-2, 2]},
    par_resolution=0.0001)

an.plot_bifurcation(show=True)
