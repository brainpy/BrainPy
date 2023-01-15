# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp

bp.math.enable_x64()


def simulation():
  model = bp.neurons.HindmarshRose(1)
  runner = bp.DSRunner(
    model,
    monitors=['V', 'y', 'z'],
    inputs=[model.input, 1.5],
  )
  runner.run(2000.)
  bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V')
  # bp.visualize.line_plot(runner.mon.ts, runner.mon.y, legend='y')
  # bp.visualize.line_plot(runner.mon.ts, runner.mon.z, legend='z')
  plt.show()


def bifurcation_analysis():
  model = bp.neurons.HindmarshRose(1)
  analyzer = bp.analysis.FastSlow2D(
    model,
    fast_vars={'V': [-3, 2], 'y': [-20., 3.]},
    slow_vars={'z': [-0.5, 3.]},
    pars_update={'I_ext': 1.5},
    resolutions={'z': 0.01},
    # options={bp.analysis.C.y_by_x_in_fy: lambda x: model.c - model.d * x * x}
  )
  analyzer.plot_bifurcation(num_rank=20)
  analyzer.plot_trajectory({'V': [1.], 'y': [1.], 'z': [1.]},
                           duration=1700,
                           plot_durations=[360, 1680])
  analyzer.show_figure()


def phase_plane_analysis():
  model = bp.neurons.HindmarshRose(1)
  for z in np.arange(0., 2.5, 0.3):
    analyzer = bp.analysis.PhasePlane2D(
      model,
      target_vars={'V': [-3, 2], 'y': [-20., 3.]},
      pars_update={'I_ext': 1.5, 'z': z},
      resolutions={'V': 0.01, 'y': 0.01},
    )
    analyzer.plot_nullcline()
    analyzer.plot_vector_field()
    fps = analyzer.plot_fixed_point(with_return=True)
    analyzer.plot_trajectory({'V': [fps[-1, 0] + 0.1],
                              'y': [fps[-1, 0] + 0.1]},
                             duration=500,
                             plot_durations=[400, 500])
    plt.title(f'z={z:.2f}')
    plt.show()
    # plt.savefig(f'data/z={z:.2f}.png')
    plt.close()


if __name__ == '__main__':
  simulation()
  bifurcation_analysis()
  phase_plane_analysis()
