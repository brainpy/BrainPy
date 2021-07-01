# -*- coding: utf-8 -*-

import brainpy as bp


class HRNeuron(bp.NeuGroup):
    def __init__(self, num, a=1., b=3., c=1., d=5., s=4., x_r=-1.6, r=0.001,
                 **kwargs):

        def dev_hr(x, y, z, t, Isyn):
            dx = y - a * x ** 3 + b * x * x - z + Isyn
            dy = c - d * x * x - y
            dz = r * (s * (x - x_r) - z)
            return dx, dy, dz

        self.int_hr = bp.odeint(f=dev_hr, method='rk4', dt=0.02)

        super(HRNeuron, self).__init__(size=num, **kwargs)

    def update(self, _t):
        pass


hr = HRNeuron(1)

analyzer = bp.analysis.FastSlowBifurcation(
    integrals=hr.int_hr,
    fast_vars={'x': [-3, 3], 'y': [-10., 5.]},
    slow_vars={'z': [-5., 5.]},
    pars_update={'Isyn': 0.5},
    numerical_resolution=0.001
)
analyzer.plot_bifurcation()
analyzer.plot_trajectory([{'x': 1., 'y': 0., 'z': -0.0}],
                         duration=300.,
                         show=True)

