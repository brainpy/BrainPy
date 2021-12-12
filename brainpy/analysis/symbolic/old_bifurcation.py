# -*- coding: utf-8 -*-

import gc

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from brainpy import errors, math
from brainpy.analysis import stability
from brainpy.analysis import utils
from brainpy.analysis.symbolic import old_base
from brainpy.analysis.utils import olds

__all__ = [
  'OldBifurcation',
  '_Bifurcation1D',
  '_Bifurcation2D',

  'OldFastSlowBifurcation',
  '_FastSlow1D',
  '_FastSlow2D',
]


class OldBifurcation(object):
  """A tool class for bifurcation analysis.

  The bifurcation analyzer is restricted to analyze the bifurcation
  relation between membrane potential and a given model parameter
  (co-dimension-1 case) or two model parameters (co-dimension-2 case).

  Externally injected current is also treated as a model parameter in
  this class, instead of a model state.

  Examples
  --------

  - Tutorials please see: `Dynamics Analysis (Symbolic) <../../../tutorial_analysis/symbolic.ipynb>`_


  Parameters
  ----------

  integrals : function, functions
      The integral functions defined with `brainpy.odeint` or
      `brainpy.sdeint` or `brainpy.ddeint`, or `brainpy.fdeint`.
  target_vars : dict
      The target dynamical variables. It must a dictionary which
      specifies the boundary of the variables: `{'var1': [min, max]}`.
  fixed_vars : dict, optional
      The fixed variables. It must a fixed value with the format of `{'var1': value}`.
  target_pars : dict, optional
      The parameters which can be dynamical varied. It must be a dictionary which
      specifies the boundary of the variables: `{'par1': [min, max]}`
  pars_update : dict, optional
      The parameters to update. Or, they can be treated as staitic parameters.
      Same with the `fixed_vars`, they are must fixed values with the format of
      `{'par1': value}`.
  numerical_resolution : float, dict, optional
      The resolution for numerical iterative solvers. Default is 0.1.
      It can set the numerical resolution of dynamical variables or dynamical parameters.
      For example,

      - set ``numerical_resolution=0.1`` will generalize it to all variables and parameters;
      - set ``numerical_resolution={var1: 0.1, var2: 0.2, par1: 0.1, par2: 0.05}`` will
        specify the particular resolutions to variables and parameters.
      - Moreover, you can also set ``numerical_resolution={var1: np.array([...]), var2: 0.1}``
        to specify the search points need to explore for variable `var1`. This will be useful
        to set sense search points at some inflection points.
  options : dict, optional
      The other setting parameters, which includes:

      - **perturbation**: float. The small perturbation used to solve the function derivatives.
      - **sympy_solver_timeout**: float, with the unit of second. The maximum time allowed
        to use sympy solver to get the variable relationship.
      - **escape_sympy_solver**: bool. Whether escape to use sympy solver, and directly use
        numerical optimization method to solve the nullcline and fixed points.
      - **lim_scale**: float. The axis limit scale factor. Default is 1.05. The setting means
        the axes will be clipped to ``[var_min * (1-lim_scale)/2, var_max * (var_max-1)/2]``.

      The parameters which are usefull for two-dimensional bifurcation analysis:

      - **shgo_args**: dict. Arguments of `shgo` optimization method, which can be used to
        set the fields of: constraints, n, iters, callback, minimizer_kwargs, options, sampling_method.
      - **show_shgo**: bool. whether print the shgo's value.
      - **fl_tol**: float. The tolerance of the function value to recognize it as a candidate of
        function root point.
      - **xl_tol**: float. The tolerance of the l2 norm distances between this point and
        previous points. If the norm distances are all bigger than `xl_tol` means this
        point belong to a new function root point.
  """

  def __init__(self, integrals, target_pars, target_vars, fixed_vars=None, pars_update=None,
               numerical_resolution=0.1, options=None):
    # check "model"
    self.model = olds.integrators_into_model(integrals)

    # check "target_pars"
    if not isinstance(target_pars, dict):
      raise errors.BrainPyError('"target_pars" must a dict with the format of: '
                                '{"Parameter A": [A_min, A_max],'
                                ' "Parameter B": [B_min, B_max]}')
    self.target_pars = target_pars
    if len(target_pars) > 2:
      raise errors.BrainPyError("The number of parameters in bifurcation"
                                "analysis cannot exceed 2.")

    # check "fixed_vars"
    if fixed_vars is None:
      fixed_vars = dict()
    if not isinstance(fixed_vars, dict):
      raise errors.BrainPyError('"fixed_vars" must be a dict the format of: '
                                '{"Variable A": A_value, "Variable B": B_value}')
    self.fixed_vars = fixed_vars

    # check "target_vars"
    if not isinstance(target_vars, dict):
      raise errors.BrainPyError('"target_vars" must a dict with the format of: '
                                '{"Variable A": [A_min, A_max], "Variable B": [B_min, B_max]}')
    self.target_vars = target_vars

    # check "pars_update"
    if pars_update is None:
      pars_update = dict()
    if not isinstance(pars_update, dict):
      raise errors.BrainPyError('"pars_update" must be a dict the format of: '
                                '{"Par A": A_value, "Par B": B_value}')
    for key in pars_update.keys():
      if (key not in self.model.scopes) and (key not in self.model.parameters):
        raise errors.BrainPyError(f'"{key}" is not a valid parameter in "{integrals}". ')
    self.pars_update = pars_update

    # bifurcation analysis
    if len(self.target_vars) == 1:
      self.analyzer = _Bifurcation1D(model_or_integrals=self.model,
                                     target_pars=target_pars,
                                     target_vars=target_vars,
                                     fixed_vars=fixed_vars,
                                     pars_update=pars_update,
                                     numerical_resolution=numerical_resolution,
                                     options=options)

    elif len(self.target_vars) == 2:
      self.analyzer = _Bifurcation2D(model_or_integrals=self.model,
                                     target_pars=target_pars,
                                     target_vars=target_vars,
                                     fixed_vars=fixed_vars,
                                     pars_update=pars_update,
                                     numerical_resolution=numerical_resolution,
                                     options=options)

    else:
      raise errors.BrainPyError(f'Cannot analyze three dimensional system: {self.target_vars}')

  def plot_bifurcation(self, *args, **kwargs):
    """Plot bifurcation, which support bifurcation analysis of
    co-dimension 1 and co-dimension 2.

    Parameters
    ----------
    show : bool
        Whether show the bifurcation figure.

    Returns
    -------
    points : dict
        The bifurcation points which specifies their fixed points
        and corresponding stability.
    """
    return self.analyzer.plot_bifurcation(*args, **kwargs)

  def plot_limit_cycle_by_sim(self, var, duration=100, inputs=(), plot_style=None, tol=0.001, show=False):
    """Plot limit cycles by the simulation results.

    This function help users plot the limit cycles through the simulation results,
    in which the periodic signals will be automatically found and then treated them
    as the candidate of limit cycles.

    Parameters
    ----------
    var : str
        The target variable to found its limit cycles.
    duration : int, float, tuple, list
        The simulation duration.
    inputs : tuple, list
        The simulation inputs.
    plot_style : dict
        The limit cycle plotting style settings.
    tol : float
        The tolerance to found periodic signals.
    show : bool
        Whether show the figure.
    """
    self.analyzer.plot_limit_cycle_by_sim(var=var, duration=duration, inputs=inputs,
                                          plot_style=plot_style, tol=tol, show=show)


class _Bifurcation1D(old_base.OldSymAnalyzer1D):
  """Bifurcation analysis of 1D system.

  Using this class, we can make co-dimension1 or co-dimension2 bifurcation analysis.
  """

  def __init__(self, model_or_integrals, target_pars, target_vars, fixed_vars=None,
               pars_update=None, numerical_resolution=0.1, options=None):
    super(_Bifurcation1D, self).__init__(model_or_integrals=model_or_integrals,
                                         target_pars=target_pars,
                                         target_vars=target_vars,
                                         fixed_vars=fixed_vars,
                                         pars_update=pars_update,
                                         numerical_resolution=numerical_resolution,
                                         options=options)

  def plot_bifurcation(self, show=False):
    utils.output('plot bifurcation ...')

    f_fixed_point = self.get_f_fixed_point()
    f_dfdx = self.get_f_dfdx()

    if len(self.target_pars) == 1:
      container = {c: {'p': [], 'x': []} for c in stability.get_1d_stability_types()}

      # fixed point
      par_a = self.target_par_names[0]
      for p in self.resolutions[par_a]:
        xs = f_fixed_point(p)
        for x in xs:
          dfdx = f_dfdx(x, p)
          fp_type = stability.stability_analysis(dfdx)
          container[fp_type]['p'].append(p)
          container[fp_type]['x'].append(x)

      # visualization
      plt.figure(self.x_var)
      for fp_type, points in container.items():
        if len(points['x']):
          plot_style = stability.plot_scheme[fp_type]
          plt.plot(points['p'], points['x'], '.', **plot_style, label=fp_type)
      plt.xlabel(par_a)
      plt.ylabel(self.x_var)

      # scale = (self.options.lim_scale - 1) / 2
      # plt.xlim(*utils.rescale(self.target_pars[self.dpar_names[0]], scale=scale))
      # plt.ylim(*utils.rescale(self.target_vars[self.x_var], scale=scale))

      plt.legend()
      if show:
        plt.show()

    elif len(self.target_pars) == 2:
      container = {c: {'p0': [], 'p1': [], 'x': []} for c in stability.get_1d_stability_types()}

      # fixed point
      for p0 in self.resolutions[self.target_par_names[0]]:
        for p1 in self.resolutions[self.target_par_names[1]]:
          xs = f_fixed_point(p0, p1)
          for x in xs:
            dfdx = f_dfdx(x, p0, p1)
            fp_type = stability.stability_analysis(dfdx)
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

      # scale = (self.options.lim_scale - 1) / 2
      # ax.set_xlim(*utils.rescale(self.target_pars[self.dpar_names[0]], scale=scale))
      # ax.set_ylim(*utils.rescale(self.target_pars[self.dpar_names[1]], scale=scale))
      # ax.set_zlim(*utils.rescale(self.target_vars[self.x_var], scale=scale))

      ax.grid(True)
      ax.legend()
      if show:
        plt.show()

    else:
      raise errors.BrainPyError(f'Cannot visualize co-dimension {len(self.target_pars)} '
                                f'bifurcation.')
    return container

  def plot_limit_cycle_by_sim(self, *args, **kwargs):
    raise NotImplementedError('1D phase plane do not support plot_limit_cycle_by_sim.')


class _Bifurcation2D(old_base.OldSymAnalyzer2D):
  """Bifurcation analysis of 2D system.

  Using this class, we can make co-dimension1 or co-dimension2 bifurcation analysis.
  """

  def __init__(self, model_or_integrals, target_pars, target_vars, fixed_vars=None,
               pars_update=None, numerical_resolution=0.1, options=None):
    super(_Bifurcation2D, self).__init__(model_or_integrals=model_or_integrals,
                                         target_pars=target_pars,
                                         target_vars=target_vars,
                                         fixed_vars=fixed_vars,
                                         pars_update=pars_update,
                                         numerical_resolution=numerical_resolution,
                                         options=options)

    self.fixed_points = None

  def plot_bifurcation(self, show=False):
    utils.output('plot bifurcation ...')

    # functions
    f_fixed_point = self.get_f_fixed_point()
    f_jacobian = self.get_f_jacobian()

    # bifurcation analysis of co-dimension 1
    if len(self.target_pars) == 1:
      container = {c: {'p': [], self.x_var: [], self.y_var: []}
                   for c in stability.get_2d_stability_types()}

      # fixed point
      for p in self.resolutions[self.target_par_names[0]]:
        xs, ys = f_fixed_point(p)
        for x, y in zip(xs, ys):
          dfdx = f_jacobian(x, y, p)
          fp_type = stability.stability_analysis(dfdx)
          container[fp_type]['p'].append(p)
          container[fp_type][self.x_var].append(x)
          container[fp_type][self.y_var].append(y)

      # visualization
      for var in self.target_var_names:
        plt.figure(var)
        for fp_type, points in container.items():
          if len(points['p']):
            plot_style = stability.plot_scheme[fp_type]
            plt.plot(points['p'], points[var], '.', **plot_style, label=fp_type)
        plt.xlabel(self.target_par_names[0])
        plt.ylabel(var)

        # scale = (self.options.lim_scale - 1) / 2
        # plt.xlim(*utils.rescale(self.target_pars[self.dpar_names[0]], scale=scale))
        # plt.ylim(*utils.rescale(self.target_vars[var], scale=scale))

        plt.legend()
      if show:
        plt.show()

    # bifurcation analysis of co-dimension 2
    elif len(self.target_pars) == 2:
      container = {c: {'p0': [], 'p1': [], self.x_var: [], self.y_var: []}
                   for c in stability.get_2d_stability_types()}

      # fixed point
      for p0 in self.resolutions[self.target_par_names[0]]:
        for p1 in self.resolutions[self.target_par_names[1]]:
          xs, ys = f_fixed_point(p0, p1)
          for x, y in zip(xs, ys):
            dfdx = f_jacobian(x, y, p0, p1)
            fp_type = stability.stability_analysis(dfdx)
            container[fp_type]['p0'].append(p0)
            container[fp_type]['p1'].append(p1)
            container[fp_type][self.x_var].append(x)
            container[fp_type][self.y_var].append(y)

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

        # scale = (self.options.lim_scale - 1) / 2
        # ax.set_xlim(*utils.rescale(self.target_pars[self.dpar_names[0]], scale=scale))
        # ax.set_ylim(*utils.rescale(self.target_pars[self.dpar_names[1]], scale=scale))
        # ax.set_zlim(*utils.rescale(self.target_vars[var], scale=scale))

        ax.grid(True)
        ax.legend()
      if show:
        plt.show()

    else:
      raise ValueError('Unknown length of parameters.')

    self.fixed_points = container
    return container

  def plot_limit_cycle_by_sim(self, var, duration=100, inputs=(), plot_style=None, tol=0.001, show=False):
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
    traj_group = olds.Trajectory(model=self.model,
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
      max_index = olds.find_indexes_of_limit_cycle_max(data, tol=tol)
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


class OldFastSlowBifurcation(object):
  """Fast slow analysis analysis proposed by John Rinzel [1]_ [2]_ [3]_.

  (J Rinzel, 1985, 1986, 1987) proposed that in a fast-slow dynamical
  system, we can treat the slow variables as the bifurcation parameters,
  and then study how the different value of slow variables affect the
  bifurcation of the fast sub-system.


  Examples
  --------

  - Tutorials please see: `Dynamics Analysis (Symbolic) <../../../tutorial_analysis/symbolic.ipynb>`_


  Parameters
  ----------

  integrals : function, functions
      The integral functions defined with `brainpy.odeint` or
      `brainpy.sdeint` or `brainpy.ddeint`, or `brainpy.fdeint`.
  fast_vars : dict
      The fast dynamical variables. It must a dictionary which
      specifies the boundary of the variables: `{'var1': [min, max]}`.
  slow_vars : dict
      The slow dynamical variables. It must a dictionary which
      specifies the boundary of the variables: `{'var1': [min, max]}`.
  fixed_vars : dict
      The fixed variables. It must a fixed value with the format of `{'var1': value}`.
  pars_update : dict, optional
      The parameters to update. Or, they can be treated as staitic parameters.
      Same with the `fixed_vars`, they are must fixed values with the format of
      `{'par1': value}`.
  numerical_resolution : float, dict
      The resolution for numerical iterative solvers. Default is 0.1.
      It can set the numerical resolution of dynamical variables or dynamical parameters.
      For example, set ``numerical_resolution=0.1`` will generalize it to all
      variables and parameters;
      set ``numerical_resolution={var1: 0.1, var2: 0.2, par1: 0.1, par2: 0.05}`` will
      specify the particular resolutions to variables and parameters.
      Moreover, you can also set
      ``numerical_resolution={var1: np.array([...]), var2: 0.1}`` to specify the
      search points need to explore for variable `var1`. This will be useful to
      set sense search points at some inflection points.
  options : dict, optional
      The other setting parameters, which includes:

          perturbation
              float. The small perturbation used to solve the function derivatives.
          sympy_solver_timeout
              float, with the unit of second. The maximum time allowed to use sympy solver
              to get the variable relationship.
          escape_sympy_solver
              bool. Whether escape to use sympy solver, and directly use numerical optimization
              method to solve the nullcline and fixed points.
          lim_scale
              float. The axis limit scale factor. Default is 1.05. The setting means
              the axes will be clipped to ``[var_min * (1-lim_scale)/2, var_max * (var_max-1)/2]``.

  References
  ----------

  .. [1] Rinzel, John. "Bursting oscillations in an excitable
         membrane model." In Ordinary and partial differential
         equations, pp. 304-316. Springer, Berlin, Heidelberg, 1985.
  .. [2] Rinzel, John , and Y. S. Lee . On Different Mechanisms for
         Membrane Potential Bursting. Nonlinear Oscillations in
         Biology and Chemistry. Springer Berlin Heidelberg, 1986.
  .. [3] Rinzel, John. "A formal classification of bursting mechanisms
         in excitable systems." In Mathematical topics in population
         biology, morphogenesis and neurosciences, pp. 267-281.
         Springer, Berlin, Heidelberg, 1987.

  """

  def __init__(self, integrals, fast_vars, slow_vars, fixed_vars=None,
               pars_update=None, numerical_resolution=0.1, options=None):
    # check "model"
    self.model = olds.integrators_into_model(integrals)

    # check "fast_vars"
    if not isinstance(fast_vars, dict):
      raise errors.BrainPyError('"fast_vars" must a dict with the format of: '
                                '{"Var A": [A_min, A_max],'
                                ' "Var B": [B_min, B_max]}')
    self.fast_vars = fast_vars
    if len(fast_vars) > 2:
      raise errors.BrainPyError("FastSlowBifurcation can only analyze the system with less "
                                "than two-variable fast subsystem.")

    # check "slow_vars"
    if not isinstance(slow_vars, dict):
      raise errors.BrainPyError('"slow_vars" must a dict with the format of: '
                                '{"Variable A": [A_min, A_max], '
                                '"Variable B": [B_min, B_max]}')
    self.slow_vars = slow_vars
    if len(slow_vars) > 2:
      raise errors.BrainPyError("FastSlowBifurcation can only analyze the system with less "
                                "than two-variable slow subsystem.")
    for key in self.slow_vars:
      self.model.variables.remove(key)
      self.model.parameters.append(key)

    # check "fixed_vars"
    if fixed_vars is None:
      fixed_vars = dict()
    if not isinstance(fixed_vars, dict):
      raise errors.BrainPyError('"fixed_vars" must be a dict the format of: '
                                '{"Variable A": A_value, "Variable B": B_value}')
    self.fixed_vars = fixed_vars

    # check "pars_update"
    if pars_update is None:
      pars_update = dict()
    if not isinstance(pars_update, dict):
      raise errors.BrainPyError('"pars_update" must be a dict the format of: '
                                '{"Par A": A_value, "Par B": B_value}')
    for key in pars_update.keys():
      if (key not in self.model.scopes) and (key not in self.model.parameters):
        raise errors.BrainPyError(f'"{key}" is not a valid parameter in "{integrals}" model. ')
    self.pars_update = pars_update

    # bifurcation analysis
    if len(self.fast_vars) == 1:
      self.analyzer = _FastSlow1D(model_or_integrals=self.model,
                                  fast_vars=fast_vars,
                                  slow_vars=slow_vars,
                                  fixed_vars=fixed_vars,
                                  pars_update=pars_update,
                                  numerical_resolution=numerical_resolution,
                                  options=options)

    elif len(self.fast_vars) == 2:
      self.analyzer = _FastSlow2D(model_or_integrals=self.model,
                                  fast_vars=fast_vars,
                                  slow_vars=slow_vars,
                                  fixed_vars=fixed_vars,
                                  pars_update=pars_update,
                                  numerical_resolution=numerical_resolution,
                                  options=options)

    else:
      raise errors.BrainPyError(f'Cannot analyze {len(fast_vars)} dimensional fast system.')

  def plot_bifurcation(self, *args, **kwargs):
    """Plot bifurcation.

    Parameters
    ----------
    show : bool
        Whether show the bifurcation figure.

    Returns
    -------
    points : dict
        The bifurcation points which specifies their fixed points
        and corresponding stability.
    """
    return self.analyzer.plot_bifurcation(*args, **kwargs)

  def plot_trajectory(self, *args, **kwargs):
    """Plot trajectory.

    This function helps users to plot specific trajectories.

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
    self.analyzer.plot_trajectory(*args, **kwargs)

  def plot_limit_cycle_by_sim(self, *args, **kwargs):
    """Plot limit cycles by the simulation results.

    This function help users plot the limit cycles through the simulation results,
    in which the periodic signals will be automatically found and then treated them
    as the candidate of limit cycles.

    Parameters
    ----------
    var : str
        The target variable to found its limit cycles.
    duration : int, float, tuple, list
        The simulation duration.
    inputs : tuple, list
        The simulation inputs.
    plot_style : dict
        The limit cycle plotting style settings.
    tol : float
        The tolerance to found periodic signals.
    show : bool
        Whether show the figure.
    """
    self.analyzer.plot_limit_cycle_by_sim(*args, **kwargs)


class _FastSlowTrajectory(object):
  def __init__(self, model_or_intgs, fast_vars, slow_vars, fixed_vars=None,
               pars_update=None, **kwargs):
    if isinstance(model_or_intgs, olds.SymbolicDynSystem):
      self.model = model_or_intgs
    elif (isinstance(model_or_intgs, (list, tuple)) and callable(model_or_intgs[0])) or callable(model_or_intgs):
      self.model = olds.integrators_into_model(model_or_intgs)
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
      traj_group = olds.Trajectory(model=self.model,
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
      start = int(plot_duration[init_i][0] / math.get_dt())
      end = int(plot_duration[init_i][1] / math.get_dt())

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


class _FastSlow1D(_Bifurcation1D):
  def __init__(self, model_or_integrals, fast_vars, slow_vars, fixed_vars=None,
               pars_update=None, numerical_resolution=0.1, options=None):
    super(_FastSlow1D, self).__init__(model_or_integrals=model_or_integrals,
                                      target_pars=slow_vars,
                                      target_vars=fast_vars,
                                      fixed_vars=fixed_vars,
                                      pars_update=pars_update,
                                      numerical_resolution=numerical_resolution,
                                      options=options)
    self.traj = _FastSlowTrajectory(model_or_intgs=model_or_integrals,
                                    fast_vars=fast_vars,
                                    slow_vars=slow_vars,
                                    fixed_vars=fixed_vars,
                                    pars_update=pars_update,
                                    numerical_resolution=numerical_resolution,
                                    options=options)

  def plot_trajectory(self, *args, **kwargs):
    self.traj.plot_trajectory(*args, **kwargs)

  def plot_bifurcation(self, *args, **kwargs):
    return super(_FastSlow1D, self).plot_bifurcation(*args, **kwargs)

  def plot_limit_cycle_by_sim(self, *args, **kwargs):
    super(_FastSlow1D, self).plot_limit_cycle_by_sim(*args, **kwargs)


class _FastSlow2D(_Bifurcation2D):
  def __init__(self, model_or_integrals, fast_vars, slow_vars, fixed_vars=None,
               pars_update=None, numerical_resolution=0.1, options=None):
    super(_FastSlow2D, self).__init__(model_or_integrals=model_or_integrals,
                                      target_pars=slow_vars,
                                      target_vars=fast_vars,
                                      fixed_vars=fixed_vars,
                                      pars_update=pars_update,
                                      numerical_resolution=numerical_resolution,
                                      options=options)
    self.traj = _FastSlowTrajectory(model_or_intgs=model_or_integrals,
                                    fast_vars=fast_vars,
                                    slow_vars=slow_vars,
                                    fixed_vars=fixed_vars,
                                    pars_update=pars_update,
                                    numerical_resolution=numerical_resolution,
                                    options=options)

  def plot_trajectory(self, *args, **kwargs):
    self.traj.plot_trajectory(*args, **kwargs)

  def plot_bifurcation(self, *args, **kwargs):
    return super(_FastSlow2D, self).plot_bifurcation(*args, **kwargs)

  def plot_limit_cycle_by_sim(self, *args, **kwargs):
    super(_FastSlow2D, self).plot_limit_cycle_by_sim(*args, **kwargs)


if __name__ == '__main__':
  Axes3D
