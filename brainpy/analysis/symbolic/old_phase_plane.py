# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import numpy as np

from brainpy import errors, math
from brainpy.analysis import stability
from brainpy.analysis.symbolic import old_base, utils
from brainpy.integrators.base import Integrator
from brainpy.simulation.brainobjects.base import DynamicalSystem

logger = logging.getLogger('brainpy.analysis.symbolic')

__all__ = [
  'OldPhasePlane',
  '_PhasePlane1D',
  '_PhasePlane2D',
]


class OldPhasePlane(object):
  """A tool class for phase plane analysis.

  `PhasePlane` is used to analyze the phase portrait of 1D
  or 2D dynamical systems. It can also be used to analyze the phase
  portrait of high-dimensional system but with the fixation of other
  variables to preserve only one/two variables dynamical.

  Examples
  --------

  - Tutorials please see: `Dynamics Analysis (Symbolic) <../../../tutorial_analysis/symbolic.ipynb>`_



  Parameters
  ----------
  model : DynamicalSystem, Integrator, list of Integrator, tuple of Integrator
      The neuron model which defines the differential equations.
  target_vars : dict
      The target variables to analyze, with the format of
      `{'var1': [var_min, var_max], 'var2': [var_min, var_max]}`.
  fixed_vars : dict, optional
      The fixed variables, which means the variables will not be updated.
  pars_update : dict, optional
      The parameters in the differential equations to update.
  numerical_resolution : float, dict, optional
      The variable resolution for numerical iterative solvers.
      This variable will be useful in the solving of nullcline and fixed points
      by using the iterative optimization method.

      - It can be a float, which will be used as ``numpy.arange(var_min, var_max, resolution)``.
      - Or, it can be a dict, with the format of ``{'var1': resolution1, 'var2': resolution2}``.
      - Or, it can be a dict with the format of ``{'var1': np.arange(x, x, x), 'var2': np.arange(x, x, x)}``.

  options : dict, optional
      The other setting parameters, which includes:

      - **lim_scale**: float. The axis limit scale factor. Default is 1.05. The setting means
        the axes will be clipped to ``[var_min * (1-lim_scale)/2, var_max * (var_max-1)/2]``.
      - **sympy_solver_timeout**: float, with the unit of second. The maximum time allowed to
        use sympy solver to get the variable relationship.
      - **escape_sympy_solver**: bool. Whether escape to use sympy solver, and directly use
        numerical optimization method to solve the nullcline and fixed points.
      - **shgo_args**: dict. Arguments of `shgo` optimization method, which can be used to
        set the fields of: constraints, n, iters, callback, minimizer_kwargs, options, sampling_method.
      - **show_shgo**: bool. whether print the shgo's value.
      - **perturbation**: float. The small perturbation used to solve the function derivative.
      - **fl_tol**: float. The tolerance of the function value to recognize it as a candidate of
        function root point.
      - **xl_tol**: float. The tolerance of the l2 norm distances between this point and previous
        points. If the norm distances are all bigger than `xl_tol` means this
        point belong to a new function root point.
  """

  def __init__(
      self,
      model,
      target_vars,
      fixed_vars=None,
      pars_update=None,
      numerical_resolution=0.1,
      options=None,
  ):
    # check "model"
    self.model = utils.integrators_into_model(model)

    # check "target_vars"
    if not isinstance(target_vars, dict):
      raise errors.BrainPyError('"target_vars" must a dict with the format of: '
                                '{"Variable A": [A_min, A_max], "Variable B": [B_min, B_max]}')
    self.target_vars = target_vars

    # check "fixed_vars"
    if fixed_vars is None:
      fixed_vars = dict()
    if not isinstance(fixed_vars, dict):
      raise errors.BrainPyError('"fixed_vars" must be a dict with the format of: '
                                '{"Variable A": A_value, "Variable B": B_value}')
    self.fixed_vars = fixed_vars

    # check "pars_update"
    if pars_update is None:
      pars_update = dict()
    if not isinstance(pars_update, dict):
      raise errors.BrainPyError('"pars_update" must be a dict with the format of: '
                                '{"Par A": A_value, "Par B": B_value}')
    for key in pars_update.keys():
      if (key not in self.model.scopes) and (key not in self.model.parameters):
        raise errors.BrainPyError(f'"{key}" is not a valid parameter in "{model}" model.')
    self.pars_update = pars_update

    # analyzer
    if len(target_vars) == 1:
      self.analyzer = _PhasePlane1D(model_or_integrals=self.model,
                                    target_vars=target_vars,
                                    fixed_vars=fixed_vars,
                                    pars_update=pars_update,
                                    numerical_resolution=numerical_resolution,
                                    options=options)
    elif len(target_vars) == 2:
      self.analyzer = _PhasePlane2D(model_or_integrals=self.model,
                                    target_vars=target_vars,
                                    fixed_vars=fixed_vars,
                                    pars_update=pars_update,
                                    numerical_resolution=numerical_resolution,
                                    options=options)
    else:
      raise errors.BrainPyError('BrainPy only support 1D/2D phase plane analysis. '
                                'Or, you can set "fixed_vars" to fix other variables, '
                                'then make 1D/2D phase plane analysis.')

  def plot_vector_field(self, *args, **kwargs):
    """Plot vector filed of a 2D/1D system."""

    self.analyzer.plot_vector_field(*args, **kwargs)

  def plot_fixed_point(self, *args, **kwargs):
    """Plot fixed points."""

    return self.analyzer.plot_fixed_point(*args, **kwargs)

  def plot_nullcline(self, *args, **kwargs):
    """Plot nullcline (only supported in 2D system).
    """

    self.analyzer.plot_nullcline(*args, **kwargs)

  def plot_trajectory(self, initials, duration, plot_duration=None, axes='v-v', show=False):
    """Plot trajectories according to the settings.

    Parameters
    ----------
    initials : list, tuple, dict
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
    plot_duration : tuple, list, optional
        The duration to plot. It can be a tuple with ``(start, end)``. It can
        also be a list of tuple ``[(start1, end1), (start2, end2)]`` to specify
        the plot duration for each initial value running.
    axes : str
        The axes to plot. It can be:

           - 'v-v'
                  Plot the trajectory in the 'x_var'-'y_var' axis.
           - 't-v'
                  Plot the trajectory in the 'time'-'var' axis.
    show : bool
        Whether show or not.
    """
    self.analyzer.plot_trajectory(initials=initials,
                                  duration=duration,
                                  plot_duration=plot_duration,
                                  axes=axes,
                                  show=show)

  def plot_limit_cycle_by_sim(self, initials, duration, tol=0.001, show=False):
    """Plot limit cycles according to the settings.

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
    show : bool
        Whether show or not.
    """
    self.analyzer.plot_limit_cycle_by_sim(initials=initials,
                                          duration=duration,
                                          tol=tol,
                                          show=show)


class _PhasePlane1D(sym_base.OldSymAnalyzer1D):
  """Phase plane analyzer for 1D system.
  """

  def plot_vector_field(self, show=False):
    """Plot the vector filed.

    Parameters
    ----------
    show : bool
        Whether show the figure.

    Returns
    -------
    results : np.ndarray
        The dx values.
    """
    logger.warning('plot vector field ...')

    # 1. Nullcline of the x variable
    try:
      y_val = self.get_f_dx()(self.resolutions[self.x_var])
    except TypeError:
      raise errors.BrainPyError('Missing variables. Please check and set missing '
                                'variables to "fixed_vars".')

    # 2. visualization
    label = f"d{self.x_var}dt"
    x_style = dict(color='lightcoral', alpha=.7, linewidth=4)
    plt.plot(self.resolutions[self.x_var], y_val, **x_style, label=label)
    plt.axhline(0)

    plt.xlabel(self.x_var)
    plt.ylabel(label)
    plt.xlim(*utils.rescale(self.target_vars[self.x_var],
                            scale=(self.options.lim_scale - 1.) / 2))
    plt.legend()
    if show:
      plt.show()
    return y_val

  def plot_fixed_point(self, show=False):
    """Plot the fixed point.

    Parameters
    ----------
    show : bool
        Whether show the figure.

    Returns
    -------
    points : np.ndarray
        The fixed points.
    """
    logger.warning('plot fixed point ...')

    # 1. functions
    f_fixed_point = self.get_f_fixed_point()
    f_dfdx = self.get_f_dfdx()

    # 2. stability analysis
    x_values = f_fixed_point()
    container = {a: [] for a in stability.get_1d_stability_types()}
    for i in range(len(x_values)):
      x = x_values[i]
      dfdx = f_dfdx(x)
      fp_type = stability.stability_analysis(dfdx)
      logger.warning(f"Fixed point #{i + 1} at {self.x_var}={x} is a {fp_type}.")
      container[fp_type].append(x)

    # 3. visualization
    for fp_type, points in container.items():
      if len(points):
        plot_style = stability.plot_scheme[fp_type]
        plt.plot(points, [0] * len(points), '.',
                 markersize=20, **plot_style, label=fp_type)
    plt.legend()
    if show:
      plt.show()

    return np.array(x_values)

  def plot_nullcline(self, resolution=0.1, show=False):
    raise NotImplementedError('1D phase plane do not support plot_nullcline.')

  def plot_trajectory(self, *args, **kwargs):
    raise NotImplementedError('1D phase plane do not support plot_trajectory.')

  def plot_limit_cycle_by_sim(self, *args, **kwargs):
    raise NotImplementedError('1D phase plane do not support plot_limit_cycle_by_sim.')


class _PhasePlane2D(sym_base.OldSymAnalyzer2D):
  """Phase plane analyzer for 2D system.
  """

  def plot_vector_field(self, plot_method='streamplot', plot_style=None, show=False):
    """Plot the vector field.

    Parameters
    ----------
    plot_method : str
        The method to plot the vector filed.
        It can be "streamplot" or "quiver".
    plot_style : dict, optional
        The style for vector filed plotting.

        - For ``plot_method="streamplot"``, it can set the keywords like "density",
          "linewidth", "color", "arrowsize". More settings please check
          https://matplotlib.org/api/_as_gen/matplotlib.pyplot.streamplot.html.
        - For ``plot_method="quiver"``, it can set the keywords like "color",
          "units", "angles", "scale". More settings please check
          https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html.

    Returns
    -------
    result : tuple
        The ``dx``, ``dy`` values.
    """
    logger.warning('plot vector field ...')

    if plot_style is None:
      plot_style = dict()

    xs = self.resolutions[self.x_var]
    ys = self.resolutions[self.y_var]
    X, Y = np.meshgrid(xs, ys)

    # dx
    try:
      dx = self.get_f_dx()(X, Y)
    except TypeError:
      raise errors.BrainPyError('Missing variables. Please check and set missing '
                                'variables to "fixed_vars".')

    # dy
    try:
      dy = self.get_f_dy()(X, Y)
    except TypeError:
      raise errors.BrainPyError('Missing variables. Please check and set missing '
                                'variables to "fixed_vars".')

    # vector field
    if plot_method == 'quiver':
      styles = dict()
      styles['units'] = plot_style.get('units', 'xy')
      if (not np.isnan(dx).any()) and (not np.isnan(dy).any()):
        speed = np.sqrt(dx ** 2 + dy ** 2)
        dx = dx / speed
        dy = dy / speed
      plt.quiver(X, Y, dx, dy, **styles)
    elif plot_method == 'streamplot':
      styles = dict()
      styles['arrowsize'] = plot_style.get('arrowsize', 1.2)
      styles['density'] = plot_style.get('density', 1)
      styles['color'] = plot_style.get('color', 'thistle')
      linewidth = plot_style.get('linewidth', None)
      if (linewidth is None) and (not np.isnan(dx).any()) and (not np.isnan(dy).any()):
        min_width = plot_style.get('min_width', 0.5)
        max_width = plot_style.get('min_width', 5.5)
        speed = np.sqrt(dx ** 2 + dy ** 2)
        linewidth = min_width + max_width * speed / speed.max()
      plt.streamplot(X, Y, dx, dy, linewidth=linewidth, **styles)
    else:
      raise ValueError(f'Unknown plot_method "{plot_method}", only supports "quiver" and "streamplot".')

    plt.xlabel(self.x_var)
    plt.ylabel(self.y_var)

    if show:
      plt.show()

    return dx, dy

  def plot_fixed_point(self, show=False):
    """Plot the fixed point and analyze its stability.

    Parameters
    ----------
    show : bool
        Whether show the figure.

    Returns
    -------
    results : tuple
        The value points.
    """
    logger.warning('plot fixed point ...')

    # function for fixed point solving
    f_fixed_point = self.get_f_fixed_point()
    x_values, y_values = f_fixed_point()

    # function for jacobian matrix
    f_jacobian = self.get_f_jacobian()

    # stability analysis
    # ------------------
    container = {a: {'x': [], 'y': []} for a in stability.get_2d_stability_types()}
    for i in range(len(x_values)):
      x = x_values[i]
      y = y_values[i]
      fp_type = stability.stability_analysis(f_jacobian(x, y))
      logger.warning(f"Fixed point #{i + 1} at {self.x_var}={x}, {self.y_var}={y} is a {fp_type}.")
      container[fp_type]['x'].append(x)
      container[fp_type]['y'].append(y)

    # visualization
    # -------------
    for fp_type, points in container.items():
      if len(points['x']):
        plot_style = stability.plot_scheme[fp_type]
        plt.plot(points['x'], points['y'], '.', markersize=20, **plot_style, label=fp_type)
    plt.legend()
    if show:
      plt.show()

    return x_values, y_values

  def plot_nullcline(self, numerical_setting=None, show=False):
    """Plot the nullcline.

    Parameters
    ----------
    numerical_setting : dict, optional
        Set the numerical method for solving nullclines.
        For each function setting, it contains the following keywords:

        - **coords**: The coordination setting, it can be 'var1-var2' (which means
          for each possible value 'var1' the optimizer method will search
          the zero root of 'var2') or 'var2-var1' (which means iterate each
          'var2' and get the optimization results of 'var1').
        - **plot**: It can be 'scatter' (default) or 'line'.

    show : bool
        Whether show the figure.

    Returns
    -------
    values : dict
        A dict with the format of ``{func1: (x_val, y_val), func2: (x_val, y_val)}``.
    """
    logger.warning('plot nullcline ...')

    if numerical_setting is None:
      numerical_setting = dict()
    x_setting = numerical_setting.get(self.x_eq_group.func_name, {})
    y_setting = numerical_setting.get(self.y_eq_group.func_name, {})
    x_coords = x_setting.get('coords', self.x_var + '-' + self.y_var)
    y_coords = y_setting.get('coords', self.x_var + '-' + self.y_var)
    x_plot_style = x_setting.get('plot', 'scatter')
    y_plot_style = y_setting.get('plot', 'scatter')

    xs = self.resolutions[self.x_var]
    ys = self.resolutions[self.y_var]

    # Nullcline of the y variable
    y_style = dict(color='cornflowerblue', alpha=.7, )
    y_by_x = self.get_y_by_x_in_y_eq()
    if y_by_x['status'] == 'sympy_success':
      try:
        y_values_in_y_eq = y_by_x['f'](xs)
      except TypeError:
        raise errors.BrainPyError('Missing variables. Please check and set missing '
                                  'variables to "fixed_vars".')
      x_values_in_y_eq = xs
      plt.plot(xs, y_values_in_y_eq, **y_style, label=f"{self.y_var} nullcline")

    else:
      x_by_y = self.get_x_by_y_in_y_eq()
      if x_by_y['status'] == 'sympy_success':
        try:
          x_values_in_y_eq = x_by_y['f'](ys)
        except TypeError:
          raise errors.BrainPyError('Missing variables. Please check and set missing '
                                    'variables to "fixed_vars".')
        y_values_in_y_eq = ys
        plt.plot(x_values_in_y_eq, ys, **y_style, label=f"{self.y_var} nullcline")
      else:
        # optimization results
        optimizer = self.get_f_optimize_y_nullcline(y_coords)
        x_values_in_y_eq, y_values_in_y_eq = optimizer()

        if x_plot_style == 'scatter':
          plt.plot(x_values_in_y_eq, y_values_in_y_eq, '.', **y_style, label=f"{self.y_var} nullcline")
        elif x_plot_style == 'line':
          plt.plot(x_values_in_y_eq, y_values_in_y_eq, **y_style, label=f"{self.y_var} nullcline")
        else:
          raise ValueError(f'Unknown plot style: {x_plot_style}')

    # Nullcline of the x variable
    x_style = dict(color='lightcoral', alpha=.7, )
    y_by_x = self.get_y_by_x_in_x_eq()
    if y_by_x['status'] == 'sympy_success':
      try:
        y_values_in_x_eq = y_by_x['f'](xs)
      except TypeError:
        raise errors.BrainPyError('Missing variables. Please check and set missing '
                                  'variables to "fixed_vars".')
      x_values_in_x_eq = xs
      plt.plot(xs, y_values_in_x_eq, **x_style, label=f"{self.x_var} nullcline")

    else:
      x_by_y = self.get_x_by_y_in_x_eq()
      if x_by_y['status'] == 'sympy_success':
        try:
          x_values_in_x_eq = x_by_y['f'](ys)
        except TypeError:
          raise errors.BrainPyError('Missing variables. Please check and set missing '
                                    'variables to "fixed_vars".')
        y_values_in_x_eq = ys
        plt.plot(x_values_in_x_eq, ys, **x_style, label=f"{self.x_var} nullcline")
      else:
        # optimization results
        optimizer = self.get_f_optimize_x_nullcline(x_coords)
        x_values_in_x_eq, y_values_in_x_eq = optimizer()

        # visualization
        if y_plot_style == 'scatter':
          plt.plot(x_values_in_x_eq, y_values_in_x_eq, '.', **x_style, label=f"{self.x_var} nullcline")
        elif y_plot_style == 'line':
          plt.plot(x_values_in_x_eq, y_values_in_x_eq, **x_style, label=f"{self.x_var} nullcline")
        else:
          raise ValueError(f'Unknown plot style: {x_plot_style}')
    # finally
    plt.xlabel(self.x_var)
    plt.ylabel(self.y_var)
    scale = (self.options.lim_scale - 1.) / 2
    plt.xlim(*utils.rescale(self.target_vars[self.x_var], scale=scale))
    plt.ylim(*utils.rescale(self.target_vars[self.y_var], scale=scale))
    plt.legend()
    if show:
      plt.show()

    return {self.x_eq_group.func_name: (x_values_in_x_eq, y_values_in_x_eq),
            self.y_eq_group.func_name: (x_values_in_y_eq, y_values_in_y_eq)}

  def plot_trajectory(self, initials, duration, plot_duration=None, axes='v-v', show=False):
    """Plot trajectories according to the settings.

    Parameters
    ----------
    initials : list, tuple, dict
        The initial value setting of the targets. It can be a tuple/list of floats to specify
        each value of dynamical variables (for example, ``(a, b)``). It can also be a
        tuple/list of tuple to specify multiple initial values (for example,
        ``[(a1, b1), (a2, b2)]``).
    duration : int, float, tuple, list
        The running duration. Same with the ``duration`` in ``NeuGroup.run()``.

        - It can be a int/float (``t_end``) to specify the same running end time,
        - Or it can be a tuple/list of int/float (``(t_start, t_end)``) to specify
          the start and end simulation time.
        - Or, it can be a list of tuple (``[(t1_start, t1_end), (t2_start, t2_end)]``)
          to specify the specific start and end simulation time for each initial value.
    plot_duration : tuple, list, optional
        The duration to plot. It can be a tuple with ``(start, end)``. It can
        also be a list of tuple ``[(start1, end1), (start2, end2)]`` to specify
        the plot duration for each initial value running.
    axes : str
        The axes to plot. It can be:

         - 'v-v': Plot the trajectory in the 'x_var'-'y_var' axis.
         - 't-v': Plot the trajectory in the 'time'-'var' axis.
    show : bool
        Whether show or not.
    """

    logger.warning('plot trajectory ...')

    if axes not in ['v-v', 't-v']:
      raise errors.BrainPyError(f'Unknown axes "{axes}", only support "v-v" and "t-v".')

    # 1. format the initial values
    if isinstance(initials, dict):
      initials = [initials]
    elif isinstance(initials, (list, tuple)):
      if isinstance(initials[0], (int, float)):
        initials = [{self.target_var_names[i]: v for i, v in enumerate(initials)}]
      elif isinstance(initials[0], dict):
        initials = initials
      elif isinstance(initials[0], (tuple, list)) and isinstance(initials[0][0], (int, float)):
        initials = [{self.target_var_names[i]: v for i, v in enumerate(init)} for init in initials]
      else:
        raise ValueError
    else:
      raise ValueError

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

      #   5.2 run the model
      traj_group.run(duration=duration[init_i], report=False, )

      #   5.3 legend
      legend = f'$traj_{init_i}$: '
      for key in self.target_var_names:
        legend += f'{key}={initial[key]}, '
      legend = legend[:-2]

      #   5.4 trajectory
      start = int(plot_duration[init_i][0] / math.get_dt())
      end = int(plot_duration[init_i][1] / math.get_dt())

      #   5.5 visualization
      if axes == 'v-v':
        lines = plt.plot(traj_group.mon[self.x_var][start: end, 0],
                         traj_group.mon[self.y_var][start: end, 0],
                         label=legend)
        utils.add_arrow(lines[0])
      else:
        plt.plot(traj_group.mon.ts[start: end],
                 traj_group.mon[self.x_var][start: end, 0],
                 label=legend + f', {self.x_var}')
        plt.plot(traj_group.mon.ts[start: end],
                 traj_group.mon[self.y_var][start: end, 0],
                 label=legend + f', {self.y_var}')

    # 6. visualization
    if axes == 'v-v':
      plt.xlabel(self.x_var)
      plt.ylabel(self.y_var)
      scale = (self.options.lim_scale - 1.) / 2
      plt.xlim(*utils.rescale(self.target_vars[self.x_var], scale=scale))
      plt.ylim(*utils.rescale(self.target_vars[self.y_var], scale=scale))
      plt.legend()
    else:
      plt.legend(title='Initial values')

    if show:
      plt.show()

  def plot_limit_cycle_by_sim(self, initials, duration, tol=0.001, show=False):
    """Plot trajectories according to the settings.

    Parameters
    ----------
    initials : list, tuple
        The initial value setting of the targets.

        - It can be a tuple/list of floats to specify each value of dynamical variables
          (for example, ``(a, b)``).
        - It can also be a tuple/list of tuple to specify multiple initial values (for
          example, ``[(a1, b1), (a2, b2)]``).
    duration : int, float, tuple, list
        The running duration. Same with the ``duration`` in ``NeuGroup.run()``.

        - It can be a int/float (``t_end``) to specify the same running end time,
        - Or it can be a tuple/list of int/float (``(t_start, t_end)``) to specify
          the start and end simulation time.
        - Or, it can be a list of tuple (``[(t1_start, t1_end), (t2_start, t2_end)]``)
          to specify the specific start and end simulation time for each initial value.
    show : bool
        Whether show or not.
    """
    logger.warning('plot limit cycle ...')

    # 1. format the initial values
    if isinstance(initials, dict):
      initials = [initials]
    elif isinstance(initials, (list, tuple)):
      if isinstance(initials[0], (int, float)):
        initials = [{self.target_var_names[i]: v for i, v in enumerate(initials)}]
      elif isinstance(initials[0], dict):
        initials = initials
      elif isinstance(initials[0], (tuple, list)) and isinstance(initials[0][0], (int, float)):
        initials = [{self.target_var_names[i]: v for i, v in enumerate(init)} for init in initials]
      else:
        raise ValueError
    else:
      raise ValueError

    # 2. format the running duration
    if isinstance(duration, (int, float)):
      duration = [(0, duration) for _ in range(len(initials))]
    elif isinstance(duration[0], (int, float)):
      duration = [duration for _ in range(len(initials))]
    else:
      assert len(duration) == len(initials)

    # 5. run the network
    for init_i, initial in enumerate(initials):
      traj_group = utils.Trajectory(model=self.model,
                                    size=1,
                                    target_vars=initial,
                                    fixed_vars=self.fixed_vars,
                                    pars_update=self.pars_update)

      #   5.2 run the model
      traj_group.run(duration=duration[init_i], report=False, )
      x_data = traj_group.mon[self.x_var][:, 0]
      y_data = traj_group.mon[self.y_var][:, 0]
      max_index = utils.find_indexes_of_limit_cycle_max(x_data, tol=tol)
      if max_index[0] != -1:
        x_cycle = x_data[max_index[0]: max_index[1]]
        y_cycle = y_data[max_index[0]: max_index[1]]
        # 5.5 visualization
        lines = plt.plot(x_cycle, y_cycle, label='limit cycle')
        utils.add_arrow(lines[0])
      else:
        logger.warning(f'No limit cycle found for initial value {initial}')

    # 6. visualization
    plt.xlabel(self.x_var)
    plt.ylabel(self.y_var)
    scale = (self.options.lim_scale - 1.) / 2
    plt.xlim(*utils.rescale(self.target_vars[self.x_var], scale=scale))
    plt.ylim(*utils.rescale(self.target_vars[self.y_var], scale=scale))
    plt.legend()

    if show:
      plt.show()


if __name__ == '__main__':
  DynamicalSystem
  Integrator
