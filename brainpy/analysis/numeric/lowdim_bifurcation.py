# -*- coding: utf-8 -*-

import gc
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import brainpy.math as bm
from brainpy import errors
from brainpy.analysis import stability, utils, constants as C
from brainpy.analysis.numeric.lowdim_analyzer import *

_file = sys.stderr

__all__ = [
  'Bifurcation1D',
  'Bifurcation2D',
  'FastSlow1D',
  'FastSlow2D',
]


class Bifurcation1D(Num1DAnalyzer):
  """Bifurcation analysis of 1D system.

  Using this class, we can make co-dimension1 or co-dimension2 bifurcation analysis.
  """

  def __init__(self, model, target_pars, target_vars, fixed_vars=None,
               pars_update=None, resolutions=None, options=None):
    super(Bifurcation1D, self).__init__(model=model,
                                        target_pars=target_pars,
                                        target_vars=target_vars,
                                        fixed_vars=fixed_vars,
                                        pars_update=pars_update,
                                        resolutions=resolutions,
                                        options=options)

    if len(self.target_pars) == 0:
      raise ValueError

  @property
  def F_vmap_dfxdx(self):
    if C.F_vmap_dfxdx not in self.analyzed_results:
      f = bm.jit(bm.vmap(bm.vector_grad(self.F_fx, argnums=0)), device=self.jit_device)
      self.analyzed_results[C.F_vmap_dfxdx] = f
    return self.analyzed_results[C.F_vmap_dfxdx]

  def plot_bifurcation(self, with_plot=True, show=False, with_return=False,
                       tol_loss=1e-7, loss_screen=None):
    utils.output('I am making bifurcation analysis ...')

    xs = self.resolutions[self.x_var]
    vps = bm.meshgrid(xs, *tuple(self.resolutions[p] for p in self.target_par_names))
    vps = tuple(jnp.moveaxis(vp.value, 0, 1).flatten() for vp in vps)
    candidates = vps[0]
    pars = vps[1:]
    fixed_points, selected_ids = self._get_fixed_points(candidates, *pars,
                                                        tol_loss=tol_loss,
                                                        loss_screen=loss_screen,
                                                        num_seg=len(xs))
    selected_pars = tuple(np.asarray(p)[selected_ids] for p in pars)
    dfxdx = np.asarray(self.F_vmap_dfxdx(jnp.asarray(fixed_points), *selected_pars))

    if len(self.target_pars) == 1:
      container = {c: {'p': [], 'x': []} for c in stability.get_1d_stability_types()}

      # fixed point
      for p, x, dx in zip(selected_pars[0], fixed_points, dfxdx):
        fp_type = stability.stability_analysis(dx)
        container[fp_type]['p'].append(p)
        container[fp_type]['x'].append(x)

      # visualization
      plt.figure(self.x_var)
      for fp_type, points in container.items():
        if len(points['x']):
          plot_style = stability.plot_scheme[fp_type]
          plt.plot(points['p'], points['x'], '.', **plot_style, label=fp_type)
      plt.xlabel(self.target_par_names[0])
      plt.ylabel(self.x_var)

      scale = (self.lim_scale - 1) / 2
      plt.xlim(*utils.rescale(self.target_pars[self.target_par_names[0]], scale=scale))
      plt.ylim(*utils.rescale(self.target_vars[self.x_var], scale=scale))

      plt.legend()
      if show:
        plt.show()

    elif len(self.target_pars) == 2:
      container = {c: {'p0': [], 'p1': [], 'x': []} for c in stability.get_1d_stability_types()}

      # fixed point
      for p0, p1, x, dx in zip(selected_pars[0], selected_pars[1], fixed_points, dfxdx):
        fp_type = stability.stability_analysis(dx)
        container[fp_type]['p0'].append(p0)
        container[fp_type]['p1'].append(p1)
        container[fp_type]['x'].append(x)

      # visualization
      fig = plt.figure(self.x_var)
      ax = fig.add_subplot(projection='3d')
      for fp_type, points in container.items():
        if len(points['x']):
          plot_style = stability.plot_scheme[fp_type]
          xs = points['p0']
          ys = points['p1']
          zs = points['x']
          ax.scatter(xs, ys, zs, **plot_style, label=fp_type)

      ax.set_xlabel(self.target_par_names[0])
      ax.set_ylabel(self.target_par_names[1])
      ax.set_zlabel(self.x_var)

      scale = (self.lim_scale - 1) / 2
      ax.set_xlim(*utils.rescale(self.target_pars[self.target_par_names[0]], scale=scale))
      ax.set_ylim(*utils.rescale(self.target_pars[self.target_par_names[1]], scale=scale))
      ax.set_zlim(*utils.rescale(self.target_vars[self.x_var], scale=scale))

      ax.grid(True)
      ax.legend()
      if show:
        plt.show()

    else:
      raise errors.BrainPyError(f'Cannot visualize co-dimension {len(self.target_pars)} '
                                f'bifurcation.')
    if with_return:
      return fixed_points, selected_pars, dfxdx


class Bifurcation2D(Num2DAnalyzer):
  """Bifurcation analysis of 2D system.

  Using this class, we can make co-dimension1 or co-dimension2 bifurcation analysis.
  """

  def __init__(self, model, target_pars, target_vars, fixed_vars=None,
               pars_update=None, resolutions=None, options=None):
    super(Bifurcation2D, self).__init__(model=model,
                                        target_pars=target_pars,
                                        target_vars=target_vars,
                                        fixed_vars=fixed_vars,
                                        pars_update=pars_update,
                                        resolutions=resolutions,
                                        options=options)

    if len(self.target_pars) == 0:
      raise ValueError

  @property
  def F_vmap_jacobian(self):
    if C.F_vmap_jacobian not in self.analyzed_results:
      f1 = lambda xy, *args: jnp.array([self.F_fx(xy[0], xy[1], *args),
                                        self.F_fy(xy[0], xy[1], *args)])
      f2 = bm.jit(bm.vmap(bm.jacobian(f1)), device=self.jit_device)
      self.analyzed_results[C.F_vmap_jacobian] = f2
    return self.analyzed_results[C.F_vmap_jacobian]

  def plot_bifurcation(self, with_plot=True, show=False, with_return=False,
                       tol_loss=1e-7, tol_unique=1e-2, loss_screen=None,
                       num_par_segments=1, num_fp_segment=1, nullcline_aux_filter=1.,
                       select_candidates='aux_rank', num_rank=100):
    """Make the bifurcation analysis.

    Parameters
    ----------
    with_plot: bool
    show: bool
    with_return: bool
    tol_loss: float
    tol_unique: float
    loss_screen: float, optional
    num_par_segments: int, sequence of int
    num_fp_segment: int
    nullcline_aux_filter: float
    select_candidates: str
      The method to select candidate fixed points.
    """
    utils.output('I am making bifurcation analysis ...')

    if select_candidates == 'fx-nullcline':
      fx_nullclines = self._get_fx_nullcline_points(num_segments=num_par_segments,
                                                    fp_aux_filter=nullcline_aux_filter)
      candidates = fx_nullclines[0]
      parameters = fx_nullclines[1:]
    elif select_candidates == 'fy-nullcline':
      fy_nullclines = self._get_fy_nullcline_points(num_segments=num_par_segments,
                                                    fp_aux_filter=nullcline_aux_filter)
      candidates = fy_nullclines[0]
      parameters = fy_nullclines[1:]
    elif select_candidates == 'nullclines':
      fx_nullclines = self._get_fx_nullcline_points(num_segments=num_par_segments,
                                                    fp_aux_filter=nullcline_aux_filter)
      fy_nullclines = self._get_fy_nullcline_points(num_segments=num_par_segments,
                                                    fp_aux_filter=nullcline_aux_filter)
      candidates = jnp.vstack([fx_nullclines[0], fy_nullclines[0]])
      parameters = [jnp.concatenate([fx_nullclines[i], fy_nullclines[i]])
                    for i in range(1, len(fy_nullclines))]
    elif select_candidates == 'aux_rank':
      assert nullcline_aux_filter > 0.
      candidates, parameters = self._get_fp_candidates(num_segments=num_par_segments,
                                                       num_rank=num_rank)
    else:
      raise ValueError
    candidates, _, parameters = self._get_fixed_points(candidates,
                                                       *parameters,
                                                       tol_loss=tol_loss,
                                                       tol_unique=tol_unique,
                                                       loss_screen=loss_screen,
                                                       num_segment=num_fp_segment)
    candidates = np.asarray(candidates)
    parameters = np.stack(tuple(np.asarray(p) for p in parameters)).T
    utils.output('I am trying to filter out duplicate fixed points ...')
    final_fps = []
    final_pars = []
    for par in np.unique(parameters, axis=0):
      ids = np.where(np.all(parameters == par, axis=1))[0]
      fps, ids2 = utils.keep_unique(candidates[ids], tol=tol_unique)
      final_fps.append(fps)
      final_pars.append(parameters[ids[ids2]])
    final_fps = np.vstack(final_fps)
    final_pars = np.vstack(final_pars)
    jacobians = np.asarray(self.F_vmap_jacobian(jnp.asarray(final_fps), *final_pars.T))
    utils.output(f'{C.prefix}Found {len(final_fps)} fixed points.')

    # bifurcation analysis of co-dimension 1
    if len(self.target_pars) == 1:
      container = {c: {'p': [], self.x_var: [], self.y_var: []}
                   for c in stability.get_2d_stability_types()}

      # fixed point
      for p, xy, J in zip(final_pars, final_fps, jacobians):
        fp_type = stability.stability_analysis(J)
        container[fp_type]['p'].append(p[0])
        container[fp_type][self.x_var].append(xy[0])
        container[fp_type][self.y_var].append(xy[1])

      # visualization
      for var in self.target_var_names:
        plt.figure(var)
        for fp_type, points in container.items():
          if len(points['p']):
            plot_style = stability.plot_scheme[fp_type]
            plt.plot(points['p'], points[var], '.', **plot_style, label=fp_type)
        plt.xlabel(self.target_par_names[0])
        plt.ylabel(var)

        scale = (self.lim_scale - 1) / 2
        plt.xlim(*utils.rescale(self.target_pars[self.target_par_names[0]], scale=scale))
        plt.ylim(*utils.rescale(self.target_vars[var], scale=scale))

        plt.legend()
      if show:
        plt.show()

    # bifurcation analysis of co-dimension 2
    elif len(self.target_pars) == 2:
      container = {c: {'p0': [], 'p1': [], self.x_var: [], self.y_var: []}
                   for c in stability.get_2d_stability_types()}

      # fixed point
      for p, xy, J in zip(final_pars, final_fps, jacobians):
        fp_type = stability.stability_analysis(J)
        container[fp_type]['p0'].append(p[0])
        container[fp_type]['p1'].append(p[1])
        container[fp_type][self.x_var].append(xy[0])
        container[fp_type][self.y_var].append(xy[1])

      # visualization
      for var in self.target_var_names:
        fig = plt.figure(var)
        ax = fig.add_subplot(projection='3d')
        for fp_type, points in container.items():
          if len(points['p0']):
            plot_style = stability.plot_scheme[fp_type]
            xs = points['p0']
            ys = points['p1']
            zs = points[var]
            ax.scatter(xs, ys, zs, **plot_style, label=fp_type)

        ax.set_xlabel(self.target_par_names[0])
        ax.set_ylabel(self.target_par_names[1])
        ax.set_zlabel(var)
        scale = (self.lim_scale - 1) / 2
        ax.set_xlim(*utils.rescale(self.target_pars[self.target_par_names[0]], scale=scale))
        ax.set_ylim(*utils.rescale(self.target_pars[self.target_par_names[1]], scale=scale))
        ax.set_zlim(*utils.rescale(self.target_vars[var], scale=scale))
        ax.grid(True)
        ax.legend()
      if show:
        plt.show()

    else:
      raise ValueError('Unknown length of parameters.')

    if with_return:
      return final_fps, final_pars, jacobians

  def plot_limit_cycle_by_sim(self, var, duration=100, inputs=(),
                              plot_style=None, tol=0.001, show=False):
    utils.output('plot limit cycle ...')

    if self.fixed_points is None:
      raise errors.AnalyzerError('Please call "plot_bifurcation()" before "plot_limit_cycle_by_sim()".')
    if plot_style is None:
      plot_style = dict()
    fmt = plot_style.pop('fmt', '.')

    if var not in [self.x_var, self.y_var]:
      raise errors.AnalyzerError()

    all_xs, all_ys, all_p0, all_p1 = [], [], [], []

    # unstable node
    unstable_node = self.fixed_points[stability.UNSTABLE_NODE_2D]
    all_xs.extend(unstable_node[self.x_var])
    all_ys.extend(unstable_node[self.y_var])
    if len(self.target_par_names) == 1:
      all_p0.extend(unstable_node['p'])
    elif len(self.target_par_names) == 2:
      all_p0.extend(unstable_node['p0'])
      all_p1.extend(unstable_node['p1'])
    else:
      raise ValueError

    # unstable focus
    unstable_focus = self.fixed_points[stability.UNSTABLE_FOCUS_2D]
    all_xs.extend(unstable_focus[self.x_var])
    all_ys.extend(unstable_focus[self.y_var])
    if len(self.target_par_names) == 1:
      all_p0.extend(unstable_focus['p'])
    elif len(self.target_par_names) == 2:
      all_p0.extend(unstable_focus['p0'])
      all_p1.extend(unstable_focus['p1'])
    else:
      raise ValueError

    # format points
    all_xs = np.array(all_xs)
    all_ys = np.array(all_ys)
    all_p0 = np.array(all_p0)
    all_p1 = np.array(all_p1)

    # fixed variables
    fixed_vars = dict()
    for key, val in self.fixed_vars.items():
      fixed_vars[key] = val
    fixed_vars[self.target_par_names[0]] = all_p0
    if len(self.target_par_names) == 2:
      fixed_vars[self.target_par_names[1]] = all_p1

    # initialize neuron group
    length = all_xs.shape[0]
    traj_group = utils.Trajectory(model=self.model,
                                  size=length,
                                  target_vars={self.x_var: all_xs, self.y_var: all_ys},
                                  fixed_vars=fixed_vars,
                                  pars_update=self.pars_update)
    traj_group.run(duration=duration)

    # find limit cycles
    limit_cycle_max = []
    limit_cycle_min = []
    # limit_cycle = []
    p0_limit_cycle = []
    p1_limit_cycle = []
    for i in range(length):
      data = traj_group.mon[var][:, i]
      max_index = utils.find_indexes_of_limit_cycle_max(data, tol=tol)
      if max_index[0] != -1:
        x_cycle = data[max_index[0]: max_index[1]]
        limit_cycle_max.append(data[max_index[1]])
        limit_cycle_min.append(x_cycle.min())
        # limit_cycle.append(x_cycle)
        p0_limit_cycle.append(all_p0[i])
        if len(self.target_par_names) == 2:
          p1_limit_cycle.append(all_p1[i])
    self.fixed_points['limit_cycle'] = {var: {'max': limit_cycle_max,
                                              'min': limit_cycle_min,
                                              # 'cycle': limit_cycle
                                              }}
    p0_limit_cycle = np.array(p0_limit_cycle)
    p1_limit_cycle = np.array(p1_limit_cycle)

    # visualization
    if len(self.target_par_names) == 2:
      self.fixed_points['limit_cycle'] = {'p0': p0_limit_cycle, 'p1': p1_limit_cycle}
      plt.figure(var)
      plt.plot(p0_limit_cycle, p1_limit_cycle, limit_cycle_max, **plot_style, label='limit cycle (max)')
      plt.plot(p0_limit_cycle, p1_limit_cycle, limit_cycle_min, **plot_style, label='limit cycle (min)')
      plt.legend()

    else:
      self.fixed_points['limit_cycle'] = {'p': p0_limit_cycle}
      if len(limit_cycle_max):
        plt.figure(var)
        plt.plot(p0_limit_cycle, limit_cycle_max, fmt, **plot_style, label='limit cycle (max)')
        plt.plot(p0_limit_cycle, limit_cycle_min, fmt, **plot_style, label='limit cycle (min)')
        plt.legend()

    if show:
      plt.show()

    del traj_group
    gc.collect()


class _FastSlowTrajectory(object):
  def __init__(self, model_or_intgs, fast_vars, slow_vars, fixed_vars=None,
               pars_update=None, **kwargs):
    if isinstance(model_or_intgs, utils.NumDSWrapper):
      self.model = model_or_intgs
    elif (isinstance(model_or_intgs, (list, tuple)) and callable(model_or_intgs[0])) or callable(model_or_intgs):
      self.model = utils.model_transform(model_or_intgs)
    else:
      raise ValueError
    self.fast_vars = fast_vars
    self.slow_vars = slow_vars
    self.fixed_vars = fixed_vars
    self.pars_update = pars_update
    options = kwargs.get('options', dict())
    if options is None:
      options = dict()
    self.lim_scale = options.get('lim_scale', 1.05)

    # fast variables
    self.fast_var_names = list(fast_vars.keys())

    # slow variables
    self.slow_var_names = list(slow_vars.keys())

  def plot_trajectory(self, initials, duration, plot_duration=None, show=False):
    """Plot trajectories according to the settings.

    Parameters
    ----------
    initials : list, tuple
        The initial value setting of the targets. It can be a tuple/list of floats to specify
        each value of dynamical variables (for example, ``(a, b)``). It can also be a
        tuple/list of tuple to specify multiple initial values (for example,
        ``[(a1, b1), (a2, b2)]``).
    duration : int, float, tuple, list
        The running duration. Same with the ``duration`` in ``NeuGroup.run()``.
        It can be a int/float (``t_end``) to specify the same running end time,
        or it can be a tuple/list of int/float (``(t_start, t_end)``) to specify
        the start and end simulation time. Or, it can be a list of tuple
        (``[(t1_start, t1_end), (t2_start, t2_end)]``) to specify the specific
        start and end simulation time for each initial value.
    plot_duration : tuple/list of tuple, optional
        The duration to plot. It can be a tuple with ``(start, end)``. It can
        also be a list of tuple ``[(start1, end1), (start2, end2)]`` to specify
        the plot duration for each initial value running.
    show : bool
        Whether show or not.
    """
    utils.output('plot trajectory ...')

    # 1. format the initial values
    all_vars = self.fast_var_names + self.slow_var_names
    if isinstance(initials, dict):
      initials = [initials]
    elif isinstance(initials, (list, tuple)):
      if isinstance(initials[0], (int, float)):
        initials = [{all_vars[i]: v for i, v in enumerate(initials)}]
      elif isinstance(initials[0], dict):
        initials = initials
      elif isinstance(initials[0], (tuple, list)) and isinstance(initials[0][0], (int, float)):
        initials = [{all_vars[i]: v for i, v in enumerate(init)} for init in initials]
      else:
        raise ValueError
    else:
      raise ValueError
    for initial in initials:
      if len(initial) != len(all_vars):
        raise errors.AnalyzerError(f'Should provide all fast-slow variables ({all_vars}) '
                                   f' initial values, but we only get initial values for '
                                   f'variables {list(initial.keys())}.')

    # 2. format the running duration
    if isinstance(duration, (int, float)):
      duration = [(0, duration) for _ in range(len(initials))]
    elif isinstance(duration[0], (int, float)):
      duration = [duration for _ in range(len(initials))]
    else:
      assert len(duration) == len(initials)

    # 3. format the plot duration
    if plot_duration is None:
      plot_duration = duration
    if isinstance(plot_duration[0], (int, float)):
      plot_duration = [plot_duration for _ in range(len(initials))]
    else:
      assert len(plot_duration) == len(initials)

    # 5. run the network
    for init_i, initial in enumerate(initials):
      traj_group = utils.Trajectory(model=self.model,
                                    size=1,
                                    target_vars=initial,
                                    fixed_vars=self.fixed_vars,
                                    pars_update=self.pars_update)
      traj_group.run(duration=duration[init_i], report=False)

      #   5.3 legend
      legend = f'$traj_{init_i}$: '
      for key in all_vars:
        legend += f'{key}={initial[key]}, '
      legend = legend[:-2]

      #   5.4 trajectory
      start = int(plot_duration[init_i][0] / bm.get_dt())
      end = int(plot_duration[init_i][1] / bm.get_dt())

      #   5.5 visualization
      for var_name in self.fast_var_names:
        s0 = traj_group.mon[self.slow_var_names[0]][start: end, 0]
        fast = traj_group.mon[var_name][start: end, 0]

        fig = plt.figure(var_name)
        if len(self.slow_var_names) == 1:
          lines = plt.plot(s0, fast, label=legend)
          utils.add_arrow(lines[0])
          # middle = int(s0.shape[0] / 2)
          # plt.arrow(s0[middle], fast[middle],
          #           s0[middle + 1] - s0[middle], fast[middle + 1] - fast[middle],
          #           shape='full')

        elif len(self.slow_var_names) == 2:
          fig.gca()
          s1 = traj_group.mon[self.slow_var_names[1]][start: end, 0]
          plt.plot(s0, s1, fast, label=legend)
        else:
          raise errors.AnalyzerError

    # 6. visualization
    for var_name in self.fast_vars.keys():
      fig = plt.figure(var_name)

      # scale = (self.lim_scale - 1.) / 2
      if len(self.slow_var_names) == 1:
        # plt.xlim(*utils.rescale(self.slow_vars[self.slow_var_names[0]], scale=scale))
        # plt.ylim(*utils.rescale(self.fast_vars[var_name], scale=scale))
        plt.xlabel(self.slow_var_names[0])
        plt.ylabel(var_name)
      elif len(self.slow_var_names) == 2:
        ax = fig.add_subplot(projection='3d')
        # ax.set_xlim(*utils.rescale(self.slow_vars[self.slow_var_names[0]], scale=scale))
        # ax.set_ylim(*utils.rescale(self.slow_vars[self.slow_var_names[1]], scale=scale))
        # ax.set_zlim(*utils.rescale(self.fast_vars[var_name], scale=scale))
        ax.set_xlabel(self.slow_var_names[0])
        ax.set_ylabel(self.slow_var_names[1])
        ax.set_zlabel(var_name)

      plt.legend()

    if show:
      plt.show()


class FastSlow1D(Bifurcation1D):
  def __init__(self, model, fast_vars, slow_vars, fixed_vars=None,
               pars_update=None, resolutions=None, options=None):
    super(FastSlow1D, self).__init__(model=model,
                                     target_pars=slow_vars,
                                     target_vars=fast_vars,
                                     fixed_vars=fixed_vars,
                                     pars_update=pars_update,
                                     resolutions=resolutions,
                                     options=options)
  #   self.traj = _FastSlowTrajectory(model_or_intgs=model,
  #                                   fast_vars=fast_vars,
  #                                   slow_vars=slow_vars,
  #                                   fixed_vars=fixed_vars,
  #                                   pars_update=pars_update,
  #                                   numerical_resolution=resolutions,
  #                                   options=options)
  #
  # def plot_trajectory(self, *args, **kwargs):
  #   self.traj.plot_trajectory(*args, **kwargs)

  # def plot_limit_cycle_by_sim(self, *args, **kwargs):
  #   super(FastSlow1D, self).plot_limit_cycle_by_sim(*args, **kwargs)


class FastSlow2D(Bifurcation2D):
  def __init__(self, model, fast_vars, slow_vars, fixed_vars=None,
               pars_update=None, numerical_resolution=0.1, options=None):
    super(FastSlow2D, self).__init__(model=model,
                                     target_pars=slow_vars,
                                     target_vars=fast_vars,
                                     fixed_vars=fixed_vars,
                                     pars_update=pars_update,
                                     resolutions=numerical_resolution,
                                     options=options)
  #   self.traj = _FastSlowTrajectory(model_or_intgs=model,
  #                                   fast_vars=fast_vars,
  #                                   slow_vars=slow_vars,
  #                                   fixed_vars=fixed_vars,
  #                                   pars_update=pars_update,
  #                                   numerical_resolution=numerical_resolution,
  #                                   options=options)
  #
  # def plot_trajectory(self, *args, **kwargs):
  #   self.traj.plot_trajectory(*args, **kwargs)

  def plot_limit_cycle_by_sim(self, *args, **kwargs):
    super(FastSlow2D, self).plot_limit_cycle_by_sim(*args, **kwargs)
