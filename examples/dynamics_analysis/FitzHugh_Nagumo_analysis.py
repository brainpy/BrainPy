# -*- coding: utf-8 -*-

import brainpy as bp
import numpy as np

bp.profile.set(dt=0.02, numerical_method='rk4')


def get_model(a=0.7, b=0.8, tau=12.5, Vth=1.9):
    state = bp.types.NeuState({'v': 0., 'w': 1., 'spike': 0., 'input': 0.})

    @bp.integrate
    def int_w(w, t, v):
        return (v + a - b * w) / tau

    @bp.integrate
    def int_v(v, t, w, Iext):
        return v - v * v * v / 3 - w + Iext

    def update(ST, _t):
        ST['w'] = int_w(ST['w'], _t, ST['v'])
        v = int_v(ST['v'], _t, ST['w'], ST['input'])
        ST['spike'] = np.logical_and(v >= Vth, ST['v'] < Vth)
        ST['v'] = v
        ST['input'] = 0.

    return bp.NeuType(name='FitzHugh_Nagumo',
                      ST=state,
                      steps=update)


neuron = get_model()


# # simulation
# group = bp.NeuGroup(neuron, 1, monitors=['v', 'w'])
# group.ST['v'] = -2.8
# group.ST['w'] = -1.8
# group.run(100., inputs=('ST.input', 0.8))
# bp.visualize.line_plot(group.mon.ts, group.mon.v, legend='v', )
# bp.visualize.line_plot(group.mon.ts, group.mon.w, legend='w', show=True)


# phase plane analysis
analyzer = bp.PhasePortraitAnalyzer(
    model=neuron,
    target_vars={'v': [-3, 3], 'w': [-3., 3.]},
    fixed_vars={'Iext': 0.8})
analyzer.plot_nullcline()
analyzer.plot_vector_filed()
analyzer.plot_fixed_point()
analyzer.plot_trajectory([(-2.8, -1.8, 100.)],
                         inputs=('ST.input', 0.8),
                         show=True)


# codimension 1 bifurcation analysis
analyzer = bp.BifurcationAnalyzer(
    model=neuron,
    target_pars={'Iext': [0., 1.]},
    dynamical_vars={'v': [-3, 3], 'w': [-3., 3.]},
    par_resolution=0.001,
)
analyzer.plot_bifurcation(plot_vars=['v'], show=True)


# codimension 2 bifurcation analysis
analyzer = bp.BifurcationAnalyzer(
    model=neuron,
    target_pars={'a': [0.5, 1.], 'Iext': [0., 1.]},
    dynamical_vars={'v': [-3, 3], 'w': [-3., 3.]},
    par_resolution=0.01,
)
analyzer.plot_bifurcation(plot_vars=['v'], show=True)
