# -*- coding: utf-8 -*-

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import sympy

from . import base
from . import solver
from . import utils
from .. import core
from .. import integration
from .. import profile
from .. import tools
from ..errors import ModelUseError

__all__ = [
    'PhasePlaneAnalyzer',
    'PhasePlane1DAnalyzer',
    'PhasePlane2DAnalyzer',
    'PhasePortraitAnalyzer',
]


class PhasePlaneAnalyzer(object):
    """Phase Portrait Analyzer.

    `PhasePlaneAnalyzer` is used to analyze the phase portrait of 1D
    or 2D dynamical systems. It can also be used to analyze the phase
    portrait of high-dimensional system but with the fixation of other
    variables to preserve only one/two dynamical variables.

    Parameters
    ----------
    model : NeuType
        The neuron model which defines the differential equations by using
        `brainpy.integrate`.
    target_vars : dict
        The target variables to analyze, with the format of
        `{'var1': [var_min, var_max], 'var2': [var_min, var_max]}`.
    fixed_vars : dict, optional
        The fixed variables, which means the variables will not be updated.
    pars_update : dict, optional
        The parameters in the differential equations to update.
    numerical_resolution : float, dict
        The variable resolution for numerical iterative solvers.
        This variable will be useful in the solving of nullcline and fixed points
        by using the iterative optimization method. It can be a float, which will
        be used as ``numpy.arange(var_min, var_max, resolution)``. Or, it can be
        a dict, with the format of ``{'var1': resolution1, 'var2': resolution2}``.
        Or, it can be a dict with the format of ``{'var1': np.arange(x, x, x),
        'var2': np.arange(x, x, x)}``.

    options : dict, optional
        The other setting parameters, which includes:

            lim_scale
                float. The axis limit scale factor. Default is 1.05. The setting means
                the axes will be clipped to ``[var_min * (1-lim_scale)/2, var_max * (var_max-1)/2]``.
            sympy_solver_timeout
                float, with the unit of second. The maximum  time allowed to use sympy solver
                to get the variable relationship.
            escape_sympy_solver
                bool. Whether escape to use sympy solver, and directly use numerical optimization
                method to solve the nullcline and fixed points.
            shgo_args
                dict. Arguments of `shgo` optimization method, which can be used to set the
                fields of: constraints, n, iters, callback, minimizer_kwargs, options,
                sampling_method.
            show_shgo
                bool. whether print the shgo's value.
            disturb
                float. The small disturb used to solve the function derivative.
            fl_tol
                float. The tolerance of the function value to recognize it as a condidate of
                function root point.
            xl_tol
                float. The tolerance of the l2 norm distances between this point and previous
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
        if not isinstance(model, core.NeuType):
            raise ModelUseError('Phase plane analysis only support neuron type model.')
        self.model = model

        # check "target_vars"
        if not isinstance(target_vars, dict):
            raise ModelUseError('"target_vars" must a dict with the format of: '
                                '{"Variable A": [A_min, A_max], "Variable B": [B_min, B_max]}')
        self.target_vars = target_vars

        # check "fixed_vars"
        if fixed_vars is None:
            fixed_vars = dict()
        if not isinstance(fixed_vars, dict):
            raise ModelUseError('"fixed_vars" must be a dict with the format of: '
                                '{"Variable A": A_value, "Variable B": B_value}')
        self.fixed_vars = fixed_vars

        # check "pars_update"
        if pars_update is None:
            pars_update = dict()
        if not isinstance(pars_update, dict):
            raise ModelUseError('"pars_update" must be a dict with the format of: '
                                '{"Par A": A_value, "Par B": B_value}')
        for key in pars_update.keys():
            if key not in model.step_scopes:
                raise ModelUseError(f'"{key}" is not a valid parameter in "{model.name}" model.')
        self.pars_update = pars_update

        # check for "options"
        if options is None:
            options = dict()
        self.options = tools.DictPlus()
        self.options['resolution'] = options.get('resolution', 0.1)
        self.options['lim_scale'] = options.get('lim_scale', 1.05)
        self.options['sympy_solver_timeout'] = options.get('sympy_solver_timeout', 5)  # s
        self.options['escape_sympy_solver'] = options.get('escape_sympy_solver', False)
        self.options['shgo_args'] = options.get('shgo_args', dict())
        self.options['show_shgo'] = options.get('show_shgo', False)
        self.options['disturb'] = options.get('disturb', 1e-4)
        self.options['fl_tol'] = options.get('fl_tol', 1e-6)
        self.options['xl_tol'] = options.get('xl_tol', 1e-4)

        # analyzer
        if len(target_vars) == 1:
            self.analyzer = PhasePlane1DAnalyzer(model=model,
                                                 target_vars=target_vars,
                                                 fixed_vars=fixed_vars,
                                                 pars_update=pars_update,
                                                 numerical_resolution=numerical_resolution,
                                                 options=self.options)
        elif len(target_vars) == 2:
            self.analyzer = PhasePlane2DAnalyzer(model=model,
                                                 target_vars=target_vars,
                                                 fixed_vars=fixed_vars,
                                                 pars_update=pars_update,
                                                 numerical_resolution=numerical_resolution,
                                                 options=self.options)
        else:
            raise ModelUseError('BrainPy only support 1D/2D phase plane analysis. '
                                'Or, you can set "fixed_vars" to fix other variables, '
                                'then make 1D/2D phase plane analysis.')

    def plot_vector_field(self, *args, **kwargs):
        """Plot vector filed of a 2D/1D system."""
        self.analyzer.plot_vector_field(*args, **kwargs)

    def plot_fixed_point(self, *args, **kwargs):
        """Plot fixed points."""
        return self.analyzer.plot_fixed_point(*args, **kwargs)

    def plot_nullcline(self, *args, **kwargs):
        """Plot nullcline (only supported in 2D system)."""
        self.analyzer.plot_nullcline(*args, **kwargs)

    def plot_trajectory(self, *args, **kwargs):
        """Plot trajectories (only supported in 2D system)."""
        self.analyzer.plot_trajectory(*args, **kwargs)


class PhasePortraitAnalyzer(PhasePlaneAnalyzer):
    def __init__(self, *args, **kwargs):
        print('PhasePortraitAnalyzer will be removed after version 0.4.0. '
              'Please use ``brainpy.PhasePlaneAnalyzer`` instead '
              'of ``brainpy.PhasePortraitAnalyzer``.')
        super(PhasePortraitAnalyzer, self).__init__(*args, **kwargs)


class PhasePlane1DAnalyzer(base.Base1DNeuronAnalyzer):
    def __init__(self, *args, **kwargs):
        super(PhasePlane1DAnalyzer, self).__init__(*args, **kwargs)

    def plot_vector_field(self, show=False):
        # Nullcline of the x variable
        try:
            y_val = self.get_f_dx()(self.resolutions[self.x_var])
        except TypeError:
            raise ModelUseError('Missing variables. Please check and set missing '
                                'variables to "fixed_vars".')

        # visualization
        label = f"d{self.x_var}dt"
        x_style = dict(color='lightcoral', alpha=.7, linewidth=4)
        plt.plot(self.resolutions[self.x_var], y_val, **x_style, label=label)
        plt.axhline(0)

        plt.xlabel(self.x_var)
        plt.ylabel(label)
        plt.xlim(*utils.rescale(self.target_vars[self.x_var], scale=(self.options.lim_scale - 1.) / 2))
        plt.legend()
        if show:
            plt.show()

    def plot_fixed_point(self, show=False):
        x_eq = integration.str2sympy(self.x_eq_group.sub_exprs[-1].code).expr
        x_group = self.target_eqs[self.x_var]

        # function scope
        scope = deepcopy(self.pars_update)
        scope.update(self.fixed_vars)
        scope.update(integration.get_mapping_scope())
        scope.update(self.x_eq_group.diff_eq.func_scope)

        sympy_failed = True
        if not self.options.escape_sympy_solver:
            try:
                # solve
                f = utils.timeout(self.options.sympy_solver_timeout)(
                    lambda: sympy.solve(x_eq, sympy.Symbol(self.x_var, real=True)))
                results = f()
                sympy_failed = False
                # function codes
                func_codes = [f'def solve_x():']
                for expr in x_group.sub_exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                return_expr = ', '.join([integration.sympy2str(expr) for expr in results])
                func_codes.append(f'return {return_expr}')

                # function
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                x_values = scope['solve_x']()
                x_values = np.array([x_values])
            except NotImplementedError:
                sympy_failed = True
            except KeyboardInterrupt:
                sympy_failed = True

        if sympy_failed:
            # function codes
            func_codes = [f'def optimizer_x({self.x_var}):']
            for expr in x_group.old_exprs[:-1]:
                func_codes.append(f'{expr.var_name} = {expr.code}')
            func_codes.append(f'return {x_group.old_exprs[-1].code}')
            optimizer = utils.jit_compile(scope, '\n  '.join(func_codes), 'optimizer_x')
            xs = self.resolutions[self.x_var]
            x_values = solver.find_root_of_1d(optimizer, xs)
            x_values = np.array(x_values)

        # differential #
        # ------------ #

        f_dfdx = self.get_f_dfdx()

        # stability analysis #
        # ------------------ #

        container = {a: [] for a in utils.get_1d_classification()}
        for i in range(len(x_values)):
            x = x_values[i]
            dfdx = f_dfdx(x)
            fp_type = utils.stability_analysis(dfdx)
            print(f"Fixed point #{i + 1} at {self.x_var}={x} is a {fp_type}.")
            container[fp_type].append(x)

        # visualization #
        # ------------- #
        for fp_type, points in container.items():
            if len(points):
                plot_style = utils.plot_scheme[fp_type]
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


class PhasePlane2DAnalyzer(base.Base2DNeuronAnalyzer):
    def __init__(self, *args, **kwargs):
        super(PhasePlane2DAnalyzer, self).__init__(*args, **kwargs)

        # runner for trajectory
        # ---------------------

        # cannot update dynamical parameters
        self.traj_group = core.NeuGroup(self.model,
                                        geometry=1,
                                        monitors=self.dvar_names,
                                        pars_update=self.pars_update)
        self.traj_group.runner = core.TrajectoryRunner(self.traj_group,
                                                       target_vars=self.dvar_names,
                                                       fixed_vars=self.fixed_vars)
        self.traj_initial = {key: val[0] for key, val in self.traj_group.ST.items()
                             if not key.startswith('_')}
        self.traj_net = core.Network(self.traj_group)

    def plot_vector_field(self, line_widths=(0.5, 5.5), show=False):
        """Plot the vector field.

        Parameters
        ----------
        line_widths :
        show

        Returns
        -------
        result : tuple
            The ``dx``, ``dy`` values.
        """
        xs = self.resolutions[self.x_var]
        ys = self.resolutions[self.y_var]
        X, Y = np.meshgrid(xs, ys)

        # dx
        try:
            dx = self.get_f_dx()(X, Y)
        except TypeError:
            raise ModelUseError('Missing variables. Please check and set missing '
                                'variables to "fixed_vars".')

        # dy
        try:
            dy = self.get_f_dy()(X, Y)
        except TypeError:
            raise ModelUseError('Missing variables. Please check and set missing '
                                'variables to "fixed_vars".')

        # vector field
        if np.isnan(dx).any() or np.isnan(dy).any():
            plt.streamplot(X, Y, dx, dy)
        else:
            speed = np.sqrt(dx ** 2 + dy ** 2)
            lw_min, lw_max = line_widths
            lw = lw_min + lw_max * speed / speed.max()
            plt.streamplot(X, Y, dx, dy, linewidth=lw, arrowsize=1.2, density=1, color='thistle')
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
        # function for fixed point solving
        f_fixed_point = self.get_f_fixed_point()
        x_values, y_values = f_fixed_point()

        # function for jacobian matrix
        f_jacobian = self.get_f_jacobian()

        # stability analysis
        # ------------------
        container = {a: {'x': [], 'y': []} for a in utils.get_2d_classification()}
        for i in range(len(x_values)):
            x = x_values[i]
            y = y_values[i]
            fp_type = utils.stability_analysis(f_jacobian(x, y))
            print(f"Fixed point #{i + 1} at {self.x_var}={x}, {self.y_var}={y} is a {fp_type}.")
            container[fp_type]['x'].append(x)
            container[fp_type]['y'].append(y)

        # visualization
        # -------------
        for fp_type, points in container.items():
            if len(points['x']):
                plot_style = utils.plot_scheme[fp_type]
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

                coords
                    The coordination setting, it can be 'var1-var2' (which means
                    for each possible value 'var1' the optimizer method will search
                    the zero root of 'var2') or 'var2-var1' (which means iterate each
                    'var2' and get the optimization results of 'var1').
                plot
                    It can be 'scatter' (default) or 'line'.

        show : bool
            Whether show the figure.

        Returns
        -------
        values : dict
            A dict with the format of ``{func1: (x_val, y_val), func2: (x_val, y_val)}``.
        """

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
                raise ModelUseError('Missing variables. Please check and set missing '
                                    'variables to "fixed_vars".')
            x_values_in_y_eq = xs
            plt.plot(xs, y_values_in_y_eq, **y_style, label=f"{self.y_var} nullcline")

        else:
            x_by_y = self.get_x_by_y_in_y_eq()
            if x_by_y['status'] == 'sympy_success':
                try:
                    x_values_in_y_eq = x_by_y['f'](ys)
                except TypeError:
                    raise ModelUseError('Missing variables. Please check and set missing '
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
                raise ModelUseError('Missing variables. Please check and set missing '
                                    'variables to "fixed_vars".')
            x_values_in_x_eq = xs
            plt.plot(xs, y_values_in_x_eq, **y_style, label=f"{self.y_var} nullcline")

        else:
            x_by_y = self.get_x_by_y_in_x_eq()
            if x_by_y['status'] == 'sympy_success':
                try:
                    x_values_in_x_eq = x_by_y['f'](ys)
                except TypeError:
                    raise ModelUseError('Missing variables. Please check and set missing '
                                        'variables to "fixed_vars".')
                y_values_in_x_eq = ys
                plt.plot(x_values_in_x_eq, ys, **y_style, label=f"{self.y_var} nullcline")
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

    def plot_trajectory(self, initials, duration, plot_duration=None, inputs=(), axes='v-v', show=False):
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
        plot_duration : tuple, list, optional
            The duration to plot. It can be a tuple with ``(start, end)``. It can
            also be a list of tuple ``[(start1, end1), (start2, end2)]`` to specify
            the plot duration for each initial value running.
        inputs : tuple, list
            The inputs to the model. Same with the ``inputs`` in ``NeuGroup.run()``
        axes : str
            The axes to plot. It can be:

                 - 'v-v'
                        Plot the trajectory in the 'x_var'-'y_var' axis.
                 - 't-v'
                        Plot the trajectory in the 'time'-'var' axis.
        show : bool
            Whether show or not.
        """

        if axes not in ['v-v', 't-v']:
            raise ModelUseError(f'Unknown axes "{axes}", only support "v-v" and "t-v".')

        # 1. format the initial values
        if isinstance(initials[0], (int, float)):
            initials = [initials, ]
        initials = np.array(initials)

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

        # 4. run the network
        for init_i, initial in enumerate(initials):
            #   4.1 set the initial value
            for key, val in self.traj_initial.items():
                self.traj_group.ST[key] = val
            for key_i, key in enumerate(self.dvar_names):
                self.traj_group.ST[key] = initial[key_i]
            for key, val in self.fixed_vars.items():
                if key in self.traj_group.ST:
                    self.traj_group.ST[key] = val

            #   4.2 run the model
            self.traj_net.run(duration=duration[init_i], inputs=inputs,
                              report=False, data_to_host=True, verbose=False)

            #   4.3 legend
            legend = 'traj, '
            for key_i, key in enumerate(self.dvar_names):
                legend += f'${key}_{init_i}$={initial[key_i]}, '
            legend = legend[:-2]

            #   4.4 trajectory
            start = int(plot_duration[init_i][0] / profile.get_dt())
            end = int(plot_duration[init_i][1] / profile.get_dt())

            #   4.5 visualization
            if axes == 'v-v':
                plt.plot(self.traj_group.mon[self.x_var][start: end, 0],
                         self.traj_group.mon[self.y_var][start: end, 0],
                         label=legend)
            else:
                plt.plot(self.traj_group.mon.ts[start: end],
                         self.traj_group.mon[self.x_var][start: end, 0],
                         label=legend + f', {self.x_var}')
                plt.plot(self.traj_group.mon.ts[start: end],
                         self.traj_group.mon[self.y_var][start: end, 0],
                         label=legend + f', {self.y_var}')

        # 5. visualization
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

