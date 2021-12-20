# -*- coding: utf-8 -*-

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import brainpy.math as bm
from brainpy import errors
from brainpy.analysis import stability, utils, constants as C
from brainpy.analysis.lowdim.lowdim_analyzer import *

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
    fixed_points, _, selected_pars = self._get_fixed_points(candidates, *pars,
                                                            tol_loss=tol_loss,
                                                            loss_screen=loss_screen,
                                                            num_seg=len(xs))
    dfxdx = np.asarray(self.F_vmap_dfxdx(jnp.asarray(fixed_points), *selected_pars))
    selected_pars = tuple(np.asarray(p) for p in pars)

    if with_plot:
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

    self._fixed_points = None

  @property
  def F_vmap_jacobian(self):
    if C.F_vmap_jacobian not in self.analyzed_results:
      f1 = lambda xy, *args: jnp.array([self.F_fx(xy[0], xy[1], *args),
                                        self.F_fy(xy[0], xy[1], *args)])
      f2 = bm.jit(bm.vmap(bm.jacobian(f1)), device=self.jit_device)
      self.analyzed_results[C.F_vmap_jacobian] = f2
    return self.analyzed_results[C.F_vmap_jacobian]

  def plot_bifurcation(self, with_plot=True, show=False, with_return=False,
                       tol_aux=1e-7, tol_unique=1e-2, tol_opt_candidate=None,
                       num_par_segments=1, num_fp_segment=1, nullcline_aux_filter=1.,
                       select_candidates='aux_rank', num_rank=100):
    """Make the bifurcation analysis.

    Parameters
    ----------
    with_plot: bool
      Whether plot the bifurcation figure.
    show: bool
      Whether show the figure.
    with_return: bool
      Whether return the computed bifurcation results.
    tol_aux: float
      The loss tolerance of auxiliary function :math:`f_{aux}` to confirm the fixed
      point. Default is 1e-7. Once :math:`f_{aux}(x_1) < \mathrm{tol\_aux}`,
      :math:`x_1` will be a fixed point.
    tol_unique: float
      The tolerance of distance between candidate fixed points to confirm they are
      the same. Default is 1e-2. If :math:`|x_1 - x_2| > \mathrm{tol\_unique}`,
      then :math:`x_1` and :math:`x_2` are unique fixed points. Otherwise,
      :math:`x_1` and :math:`x_2` will be treated as a same fixed point.
    tol_opt_candidate: float, optional
      The tolerance of auxiliary function :math:`f_{aux}` to select candidate
      initial points for fixed point optimization.
    num_par_segments: int, sequence of int
      How to segment parameters.
    num_fp_segment: int
      How to segment fixed points.
    nullcline_aux_filter: float
      The
    select_candidates: str
      The method to select candidate fixed points. It can be:

      - ``fx-nullcline``: use the points of fx-nullcline.
      - ``fy-nullcline``: use the points of fy-nullcline.
      - ``nullclines``: use the points in both of fx-nullcline and fy-nullcline.
      - ``aux_rank``: use the minimal value of points for the auxiliary function.
    num_rank: int
      The number of candidates to be used to optimize the fixed points.
      rank to use.

    Returns
    -------
    results : tuple
      Return a tuple of analyzed results:

      - fixed points: a 2D matrix with the shape of (num_point, num_var)
      - parameters: a 2D matrix with the shape of (num_point, num_par)
      - jacobians: a 3D tensors with the shape of (num_point, 2, 2)
    """
    utils.output('I am making bifurcation analysis ...')

    if self._can_convert_to_one_eq():
      if self.convert_type() == C.x_by_y:
        X = self.resolutions[self.y_var].value
      else:
        X = self.resolutions[self.x_var].value
      pars = tuple(self.resolutions[p].value for p in self.target_par_names)
      mesh_values = jnp.meshgrid(*((X,) + pars))
      mesh_values = tuple(jnp.moveaxis(v, 0, 1).flatten() for v in mesh_values)
      candidates = mesh_values[0]
      parameters = mesh_values[1:]

    else:
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
        candidates, parameters = self._get_fp_candidates_by_aux_rank(num_segments=num_par_segments,
                                                                     num_rank=num_rank)
      else:
        raise ValueError
    candidates, _, parameters = self._get_fixed_points(candidates,
                                                       *parameters,
                                                       tol_aux=tol_aux,
                                                       tol_unique=tol_unique,
                                                       tol_opt_candidate=tol_opt_candidate,
                                                       num_segment=num_fp_segment)
    candidates = np.asarray(candidates)
    parameters = np.stack(tuple(np.asarray(p) for p in parameters)).T
    utils.output('I am trying to filter out duplicate fixed points ...')
    final_fps = []
    final_pars = []
    # TODO: this is time consuming with many parameters
    for par in np.unique(parameters, axis=0):
      ids = np.where(np.all(parameters == par, axis=1))[0]
      fps, ids2 = utils.keep_unique(candidates[ids], tol=tol_unique)
      final_fps.append(fps)
      final_pars.append(parameters[ids[ids2]])
    final_fps = np.vstack(final_fps)  # with the shape of (num_point, num_var)
    final_pars = np.vstack(final_pars)  # with the shape of (num_point, num_par)
    jacobians = np.asarray(self.F_vmap_jacobian(jnp.asarray(final_fps), *final_pars.T))
    utils.output(f'{C.prefix}Found {len(final_fps)} fixed points.')

    # remember the fixed points for later limit cycle plotting
    self._fixed_points = (final_fps, final_pars)

    if with_plot:
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

  def plot_limit_cycle_by_sim(self, duration=100, with_plot=True, with_return=False,
                              plot_style=None, tol=0.001, show=False, dt=None, offset=1.):
    utils.output('I am plotting limit cycle ...')
    if self._fixed_points is None:
      utils.output('No fixed points found, you may call "plot_bifurcation(with_plot=True)" first.')
      return

    final_fps, final_pars = self._fixed_points
    dt = bm.get_dt() if dt is None else dt
    traject_model = utils.TrajectModel(
      initial_vars={self.x_var: final_fps[:, 0] + offset, self.y_var: final_fps[:, 1] + offset},
      integrals=[self.F_int_x, self.F_int_y],
      pars={p: v for p, v in zip(self.target_par_names, final_pars.T)},
      dt=dt
    )
    mon_res = traject_model.run(duration=duration)

    # find limit cycles
    vs_limit_cycle = tuple({'min': [], 'max': []} for _ in self.target_var_names)
    ps_limit_cycle = tuple([] for _ in self.target_par_names)
    for i in range(mon_res[self.x_var].shape[1]):
      data = mon_res[self.x_var][:, i]
      max_index = utils.find_indexes_of_limit_cycle_max(data, tol=tol)
      if max_index[0] != -1:
        cycle = data[max_index[0]: max_index[1]]
        vs_limit_cycle[0]['max'].append(mon_res[self.x_var][max_index[1], i])
        vs_limit_cycle[0]['min'].append(cycle.min())
        cycle = mon_res[self.y_var][max_index[0]: max_index[1], i]
        vs_limit_cycle[1]['max'].append(mon_res[self.y_var][max_index[1], i])
        vs_limit_cycle[1]['min'].append(cycle.min())
        for j in range(len(self.target_par_names)):
          ps_limit_cycle[j].append(final_pars[i, j])
    vs_limit_cycle = tuple({k: np.asarray(v) for k, v in lm.items()} for lm in vs_limit_cycle)
    ps_limit_cycle = tuple(np.array(p) for p in ps_limit_cycle)

    # visualization
    if with_plot:
      if plot_style is None: plot_style = dict()
      fmt = plot_style.pop('fmt', '.')

      if len(self.target_par_names) == 2:
        for i, var in enumerate(self.target_var_names):
          plt.figure(var)
          plt.plot(ps_limit_cycle[0], ps_limit_cycle[1], vs_limit_cycle[i]['max'],
                   **plot_style, label='limit cycle (max)')
          plt.plot(ps_limit_cycle[0], ps_limit_cycle[1], vs_limit_cycle[i]['min'],
                   **plot_style, label='limit cycle (min)')
          plt.legend()

      elif len(self.target_par_names) == 1:
        for i, var in enumerate(self.target_var_names):
          plt.figure(var)
          plt.plot(ps_limit_cycle[0], vs_limit_cycle[i]['max'], fmt,
                   **plot_style, label='limit cycle (max)')
          plt.plot(ps_limit_cycle[0], vs_limit_cycle[i]['min'], fmt,
                   **plot_style, label='limit cycle (min)')
          plt.legend()

      else:
        raise errors.AnalyzerError

      if show:
        plt.show()

    if with_return:
      return vs_limit_cycle, ps_limit_cycle


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


class FastSlow2D(Bifurcation2D):
  def __init__(self, model, fast_vars, slow_vars, fixed_vars=None,
               pars_update=None, resolutions=0.1, options=None):
    super(FastSlow2D, self).__init__(model=model,
                                     target_pars=slow_vars,
                                     target_vars=fast_vars,
                                     fixed_vars=fixed_vars,
                                     pars_update=pars_update,
                                     resolutions=resolutions,
                                     options=options)
