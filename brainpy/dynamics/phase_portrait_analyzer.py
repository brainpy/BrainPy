# -*- coding: utf-8 -*-

import typing
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import sympy
from numba import njit

from .solver import find_root_of_1d
from .solver import find_root_of_2d
from .utils import get_1d_classification
from .utils import get_2d_classification
from .utils import plot_scheme
from .utils import rescale
from .utils import stability_analysis
from .utils import timeout
from .. import profile
from .. import tools
from ..core import NeuType
from ..core.neurons import NeuGroup
from ..core.runner import TrajectoryRunner
from ..errors import ModelUseError
from ..integration import sympy_tools

__all__ = [
    'PhasePortraitAnalyzer',
]


class PhasePortraitAnalyzer(object):
    """Phase Portrait Analyzer.

    `PhasePortraitAnalyzer` is used to analyze the phase portrait of 1D
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
    options : dict, optional
        The other setting parameters, which includes:

            - lim_scale : float. The axis limit scale factor. Default is 1.05.
                          The setting means the axes will be clipped to
                          `[var_min * (1-lim_scale)/2, var_max * (var_max-1)/2]`.
            - sympy_solver_timeout : float, with the unit of second. The maximum 
                                     time allowed to use sympy solver to get the 
                                     variable relationship.
            - escape_sympy_solver : bool. Whether escape to use sympy solver, and
                                    directly use numerical optimization method to
                                    solve the nullcline and fixed points.
            - shgo_args : arguments of `shgo` optimization method, which can be used
                          to set the fields of: args, constraints, n, iters, callback,
                          minimizer_kwargs, options, sampling_method
            - disturb : float. The small disturb used to solve the function derivative.
            - fl_tol : float. The tolerance of the function value to recognize it as a
                       condidate of function root point.
            - xl_tol : float. The tolerance of the l2 norm distances between this point
                       and previous points. If the norm distances are all bigger than
                       `xl_tol` means this point belong to a new function root point.
    """

    def __init__(
            self,
            model,
            target_vars,
            fixed_vars=None,
            pars_update=None,
            options=None,
    ):

        # check "model"
        if not isinstance(model, NeuType):
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
        self.options['lim_scale'] = options.get('lim_scale', 1.05)
        self.options['sympy_solver_timeout'] = options.get('sympy_solver_timeout', 5)  # s
        self.options['escape_sympy_solver'] = options.get('escape_sympy_solver', False)
        self.options['shgo_args'] = options.get('shgo_args', dict())
        self.options['disturb'] = options.get('disturb', 1e-4)
        self.options['fl_tol'] = options.get('fl_tol', 1e-6)
        self.options['xl_tol'] = options.get('xl_tol', 1e-4)

        # analyzer
        if len(target_vars) == 1:
            self.analyzer = _1DSystemAnalyzer(model=model,
                                              target_vars=target_vars,
                                              fixed_vars=fixed_vars,
                                              pars_update=pars_update,
                                              options=self.options)
        elif len(target_vars) == 2:
            self.analyzer = _2DSystemAnalyzer(model=model,
                                              target_vars=target_vars,
                                              fixed_vars=fixed_vars,
                                              pars_update=pars_update,
                                              options=self.options)
        else:
            raise ModelUseError('BrainPy only support 1D/2D phase plane analysis. '
                                'Or, you can set "fixed_vars" to fix other variables, '
                                'then make 1D/2D phase plane analysis.')

    def plot_vector_field(self, resolution=0.1, line_widths=(0.5, 5.5), show=False):
        self.analyzer.plot_vector_field(resolution=resolution, line_widths=line_widths, show=show)

    def plot_fixed_point(self, resolution=0.1, show=False):
        self.analyzer.plot_fixed_point(resolution=resolution, show=show)

    def plot_nullcline(self, resolution=0.1, show=False):
        self.analyzer.plot_nullcline(resolution=resolution, show=show)

    def plot_trajectory(self, target_setting, axes='v-v', inputs=(), show=False):
        self.analyzer.plot_trajectory(target_setting, axes=axes, inputs=inputs, show=show)


class _PPAnalyzer(object):
    def __init__(self,
                 model,
                 target_vars,
                 fixed_vars=None,
                 pars_update=None,
                 options=None):
        if options is None:
            options = tools.DictPlus()
        self.options = options
        self.model = model
        self.target_vars = target_vars
        self.pars_update = pars_update

        # check "fixed_vars"
        self.fixed_vars = dict()
        for integrator in model.integrators:
            var_name = integrator.diff_eq.var_name
            if var_name not in target_vars:
                if var_name in fixed_vars:
                    self.fixed_vars[var_name] = fixed_vars.get(var_name)
                else:
                    self.fixed_vars[var_name] = model.variables.get(var_name)
        for key in fixed_vars.keys():
            if key not in self.fixed_vars:
                self.fixed_vars[key] = fixed_vars.get(key)

        # dynamical variables
        var2eq = {integrator.diff_eq.var_name: integrator for integrator in model.integrators}
        self.target_eqs = tools.DictPlus()
        for key in self.target_vars.keys():
            if key not in var2eq:
                raise ModelUseError(f'target "{key}" is not a dynamical variable.')
            integrator = var2eq[key]
            diff_eq = integrator.diff_eq
            sub_exprs = diff_eq.get_f_expressions(substitute_vars=list(self.target_vars.keys()))
            old_exprs = diff_eq.get_f_expressions(substitute_vars=None)
            self.target_eqs[key] = tools.DictPlus(sub_exprs=sub_exprs,
                                                  old_exprs=old_exprs,
                                                  diff_eq=diff_eq,
                                                  func_name=diff_eq.func_name)


class _1DSystemAnalyzer(_PPAnalyzer):
    def __init__(self, *args, **kwargs):
        super(_1DSystemAnalyzer, self).__init__(*args, **kwargs)
        self.x_var = list(self.target_vars.keys())[0]
        self.f_dfdx = None
        self.f_dx = None

    def get_f_dx(self):
        if self.f_dx is None:
            eqs_of_x = self.target_eqs[self.x_var]
            scope = deepcopy(self.pars_update)
            scope.update(self.fixed_vars)
            scope.update(sympy_tools.get_mapping_scope())
            scope.update(eqs_of_x.diff_eq.func_scope)
            func_code = f'def func({self.x_var}):\n'
            for expr in eqs_of_x.old_exprs[:-1]:
                func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {eqs_of_x.old_exprs[-1].code}'
            exec(compile(func_code, '', 'exec'), scope)
            func = scope['func']
            self.f_dx = func
        return self.f_dx

    def get_f_dfdx(self):
        if self.f_dfdx is None:
            x_symbol = sympy.Symbol(self.x_var, real=True)
            x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].sub_exprs[-1].code)
            x_eq_group = self.target_eqs[self.x_var]

            eq_x_scope = deepcopy(self.pars_update)
            eq_x_scope.update(self.fixed_vars)
            eq_x_scope.update(sympy_tools.get_mapping_scope())
            eq_x_scope.update(x_eq_group['diff_eq'].func_scope)

            # dfxdx
            try:
                f = timeout(self.options.sympy_solver_timeout)(lambda: sympy.diff(x_eq, x_symbol))
                dfxdx_expr = f()
                func_codes = [f'def dfxdx({self.x_var}):']
                for expr in x_eq_group.sub_exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfxdx_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                dfxdx = eq_x_scope['dfxdx']
            except:
                scope = dict(fx=self.get_f_dx())
                func_codes = [f'def dfxdx({self.x_var}):']
                func_codes.append(f'origin = fx({self.x_var}))')
                func_codes.append(f'disturb = fx({self.x_var}+1e-4))')
                func_codes.append(f'return (disturb - origin) / 1e-4')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfxdx = scope['dfxdx']
            self.f_dfdx = dfxdx
        return self.f_dfdx

    def plot_vector_field(self, resolution=0.1, line_widths=(0.5, 5.5), show=False):
        x = np.arange(*self.target_vars[self.x_var], resolution)
        x_style = dict(color='lightcoral', alpha=.7, linewidth=4)

        # Nullcline of the x variable
        func = self.get_f_dx()
        try:
            y_val = func(x)
        except TypeError:
            raise ModelUseError('Missing variables. Please check and set missing '
                                'variables to "fixed_vars".')

        label = f"d{self.x_var}dt"
        plt.plot(x, y_val, **x_style, label=label)
        plt.axhline(0)

        plt.xlabel(self.x_var)
        plt.ylabel(label)
        plt.xlim(*rescale(self.target_vars[self.x_var], scale=(self.options.lim_scale - 1.) / 2))
        plt.legend()
        if show:
            plt.show()

    def plot_fixed_point(self, resolution=0.1, show=False):
        x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].sub_exprs[-1].code)
        x_group = self.target_eqs[self.x_var]

        # function scope
        scope = deepcopy(self.pars_update)
        scope.update(self.fixed_vars)
        scope.update(sympy_tools.get_mapping_scope())
        scope.update(x_group.diff_eq.func_scope)

        sympy_failed = True
        if not self.options.escape_sympy_solver:
            try:
                # solve
                f = timeout(self.options.sympy_solver_timeout)(
                    lambda: sympy.solve(x_eq, sympy.Symbol(self.x_var, real=True)))
                results = f()
                sympy_failed = False
                # function codes
                func_codes = [f'def solve_x():']
                for expr in x_group.sub_exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                return_expr = ', '.join([sympy_tools.sympy2str(expr) for expr in results])
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
            exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
            optimizer = scope['optimizer_x']
            optimizer = njit(optimizer)
            x_range = np.arange(*self.target_vars[self.x_var], resolution)
            x_values = find_root_of_1d(optimizer, x_range)
            x_values = np.array(x_values)

        # differential #
        # ------------ #

        f_dfdx = self.get_f_dfdx()

        # stability analysis #
        # ------------------ #

        container = {a: [] for a in get_1d_classification()}
        for i in range(len(x_values)):
            x = x_values[i]
            dfdx = f_dfdx(x)
            fp_type = stability_analysis(dfdx)
            print(f"Fixed point #{i + 1} at {self.x_var}={x} is a {fp_type}.")
            container[fp_type].append(x)

        # visualization #
        # ------------- #

        for fp_type, points in container.items():
            if len(points):
                plot_style = plot_scheme[fp_type]
                plt.plot(points, [0] * len(points), '.', markersize=20, **plot_style, label=fp_type)

        plt.legend()
        if show:
            plt.show()

        return x_values

    def plot_nullcline(self, resolution=0.1, show=False):
        raise NotImplementedError('1D phase plane do not support plot_nullcline.')

    def plot_trajectory(self, target_setting, axes='v-v', inputs=(), show=False):
        raise NotImplementedError('1D phase plane do not support plot_trajectory.')


class _2DSystemAnalyzer(_PPAnalyzer):
    def __init__(self, *args, **kwargs):
        super(_2DSystemAnalyzer, self).__init__(*args, **kwargs)

        # get the variables `x` and `y`
        if isinstance(self.target_vars, OrderedDict):
            self.x_var, self.y_var = list(self.target_vars.keys())
        else:
            self.x_var, self.y_var = list(sorted(self.target_vars.keys()))

        self.x_by_y_in_x_eq = None  # solve x_eq to get x_by_y
        self.y_by_x_in_x_eq = None  # solve x_eq to get y_by_x
        self.x_by_y_in_y_eq = None  # solve y_eq to get x_by_y
        self.y_by_x_in_y_eq = None  # solve y_eq to get y_by_x
        self.f_dx = None  # derivative function of "x" variable
        self.f_dy = None  # derivative function of "y" variable
        self.f_jacobian = None  # function to get jacobian matrix

    def get_f_dx(self):
        """Get the derivative function of :math:`x`.
        """
        if self.f_dx is None:
            eqs_of_x = self.target_eqs[self.x_var]
            scope = deepcopy(self.pars_update)
            scope.update(self.fixed_vars)
            scope.update(sympy_tools.get_mapping_scope())
            scope.update(eqs_of_x.diff_eq.func_scope)
            func_code = f'def func({self.x_var}, {self.y_var}):\n'
            for expr in eqs_of_x.old_exprs[:-1]:
                func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {eqs_of_x.old_exprs[-1].code}'
            exec(compile(func_code, '', 'exec'), scope)
            func = scope['func']
            self.f_dx = func
        return self.f_dx

    def get_f_dy(self):
        """Get the derivative function of :math:`y`.
        """
        if self.f_dy is None:
            eqs_of_y = self.target_eqs[self.y_var]
            scope = deepcopy(self.pars_update)
            scope.update(self.fixed_vars)
            scope.update(sympy_tools.get_mapping_scope())
            scope.update(eqs_of_y.diff_eq.func_scope)
            func_code = f'def func({self.x_var}, {self.y_var}):\n'
            for expr in eqs_of_y.old_exprs[:-1]:
                func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {eqs_of_y.old_exprs[-1].code}'
            exec(compile(func_code, '', 'exec'), scope)
            self.f_dy = scope['func']
        return self.f_dy

    def get_f_jacobian(self):
        """Get the function to solve jacobian matrix.
        """
        if self.f_jacobian is None:
            x_symbol = sympy.Symbol(self.x_var, real=True)
            y_symbol = sympy.Symbol(self.y_var, real=True)
            x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].sub_exprs[-1].code)
            y_eq = sympy_tools.str2sympy(self.target_eqs[self.y_var].sub_exprs[-1].code)
            x_eq_group = self.target_eqs[self.x_var]
            y_eq_group = self.target_eqs[self.y_var]

            eq_y_scope = deepcopy(self.pars_update)
            eq_y_scope.update(self.fixed_vars)
            eq_y_scope.update(sympy_tools.get_mapping_scope())
            eq_y_scope.update(y_eq_group['diff_eq'].func_scope)

            eq_x_scope = deepcopy(self.pars_update)
            eq_x_scope.update(self.fixed_vars)
            eq_x_scope.update(sympy_tools.get_mapping_scope())
            eq_x_scope.update(x_eq_group['diff_eq'].func_scope)

            # dfxdx
            try:
                f = timeout(self.options.sympy_solver_timeout)(lambda: sympy.diff(x_eq, x_symbol))
                dfxdx_expr = f()
                func_codes = [f'def dfxdx({self.x_var}, {self.y_var}):']
                for expr in x_eq_group.sub_exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfxdx_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                dfxdx = eq_x_scope['dfxdx']
            except KeyboardInterrupt:
                scope = dict(fx=self.get_f_dx())
                func_codes = [f'def dfxdx({self.x_var}, {self.y_var}):']
                func_codes.append(f'origin = fx({self.x_var}, {self.y_var}))')
                func_codes.append(f'disturb = fx({self.x_var}+{self.options.disturb}, {self.y_var}))')
                func_codes.append(f'return (disturb - origin) / {self.options.disturb}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfxdx = scope['dfxdx']

            # dfxdy
            try:
                f = timeout(self.options.sympy_solver_timeout)(lambda: sympy.diff(x_eq, y_symbol))
                dfxdy_expr = f()
                func_codes = [f'def dfxdy({self.x_var}, {self.y_var}):']
                for expr in x_eq_group.sub_exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfxdy_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                dfxdy = eq_x_scope['dfxdy']
            except KeyboardInterrupt:
                scope = dict(fx=self.get_f_dx())
                func_codes = [f'def dfxdy({self.x_var}, {self.y_var}):']
                func_codes.append(f'origin = fx({self.x_var}, {self.y_var}))')
                func_codes.append(f'disturb = fx({self.x_var}, {self.y_var}+{self.options.disturb}))')
                func_codes.append(f'return (disturb - origin) / {self.options.disturb}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfxdy = scope['dfxdy']

            # dfydx
            try:
                f = timeout(self.options.sympy_solver_timeout)(lambda: sympy.diff(y_eq, x_symbol))
                dfydx_expr = f()
                func_codes = [f'def dfydx({self.x_var}, {self.y_var}):']
                for expr in y_eq_group.sub_exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfydx_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
                dfydx = eq_y_scope['dfydx']
            except KeyboardInterrupt:
                scope = dict(fy=self.get_f_dy())
                func_codes = [f'def dfydx({self.x_var}, {self.y_var}):']
                func_codes.append(f'origin = fy({self.x_var}, {self.y_var}))')
                func_codes.append(f'disturb = fy({self.x_var}+{self.options.disturb}, {self.y_var}))')
                func_codes.append(f'return (disturb - origin) / {self.options.disturb}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfydx = scope['dfydx']

            # dfydy
            try:
                f = timeout(self.options.sympy_solver_timeout)(lambda: sympy.diff(y_eq, y_symbol))
                dfydy_expr = f()
                func_codes = [f'def dfydy({self.x_var}, {self.y_var}):']
                for expr in y_eq_group.sub_exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfydy_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
                dfydy = eq_y_scope['dfydy']
            except KeyboardInterrupt:
                scope = dict(fy=self.get_f_dy())
                func_codes = [f'def dfydy({self.x_var}, {self.y_var}):']
                func_codes.append(f'origin = fy({self.x_var}, {self.y_var}))')
                func_codes.append(f'disturb = fy({self.x_var}, {self.y_var}+{self.options.disturb}))')
                func_codes.append(f'return (disturb - origin) / {self.options.disturb}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfydy = scope['dfydy']

            # jacobian matrix
            scope = dict(f_dfydy=dfydy, f_dfydx=dfydx, f_dfxdy=dfxdy, f_dfxdx=dfxdx, np=np)
            func_codes = [f'def f_jacobian({self.x_var}, {self.y_var}):']
            func_codes.append(f'dfxdx = f_dfxdx({self.x_var}, {self.y_var})')
            func_codes.append(f'dfxdy = f_dfxdy({self.x_var}, {self.y_var})')
            func_codes.append(f'dfydx = f_dfydx({self.x_var}, {self.y_var})')
            func_codes.append(f'dfydy = f_dfydy({self.x_var}, {self.y_var})')
            func_codes.append('return np.array([[dfxdx, dfxdy], [dfydx, dfydy]])')
            exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
            self.f_jacobian = scope['f_jacobian']

        return self.f_jacobian

    def plot_vector_field(self, resolution=0.1, line_widths=(0.5, 5.5), show=False):
        X = np.arange(*self.target_vars[self.x_var], resolution)
        Y = np.arange(*self.target_vars[self.y_var], resolution)
        X, Y = np.meshgrid(X, Y)

        # dy
        func = self.get_f_dy()
        try:
            dy = func(X, Y)
        except TypeError:
            raise ModelUseError('Missing variables. Please check and set missing '
                                'variables to "fixed_vars".')

        # dx
        func = self.get_f_dx()
        try:
            dx = func(X, Y)
        except TypeError:
            raise ModelUseError('Missing variables. Please check and set missing '
                                'variables to "fixed_vars".')

        # vector field
        speed = np.sqrt(dx ** 2 + dy ** 2)
        lw_min, lw_max = line_widths
        lw = lw_min + lw_max * speed / speed.max()
        plt.streamplot(X, Y, dx, dy, linewidth=lw, arrowsize=1.2, density=1, color='thistle')
        plt.xlabel(self.x_var)
        plt.ylabel(self.y_var)

        if show:
            plt.show()

    def plot_fixed_point(self, resolution=0.1, show=False):
        x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].sub_exprs[-1].code)
        y_eq = sympy_tools.str2sympy(self.target_eqs[self.y_var].sub_exprs[-1].code)
        x_eq_group = self.target_eqs[self.x_var]
        y_eq_group = self.target_eqs[self.y_var]

        f_get_y_by_x = None
        f_get_x_by_y = None
        can_substitute_x_group_to_y_group = False
        can_substitute_y_group_to_x_group = False

        eq_y_scope = deepcopy(self.pars_update)
        eq_y_scope.update(self.fixed_vars)
        eq_y_scope.update(sympy_tools.get_mapping_scope())
        eq_y_scope.update(y_eq_group['diff_eq'].func_scope)

        eq_x_scope = deepcopy(self.pars_update)
        eq_x_scope.update(self.fixed_vars)
        eq_x_scope.update(sympy_tools.get_mapping_scope())
        eq_x_scope.update(x_eq_group['diff_eq'].func_scope)

        timeout_len = self.options.sympy_solver_timeout

        # solve y_equations #
        # ------------------

        # 1. try to solve `f(x, y) = 0` to `y = h(x)`
        if not self.options.escape_sympy_solver:
            try:
                if self.y_by_x_in_y_eq is None:
                    print(f'SymPy solve "f({self.x_var}, {self.y_var}) = 0" to "{self.y_var} = h({self.x_var})", ',
                          end='')
                    f = timeout(timeout_len)(lambda: sympy.solve(y_eq, sympy.Symbol(self.y_var, real=True)))
                    y_by_x_in_y_eq = f()
                    if len(y_by_x_in_y_eq) > 1:
                        raise NotImplementedError('Do not support multiple values.')
                    y_by_x_in_y_eq = sympy_tools.sympy2str(y_by_x_in_y_eq[0])
                    self.y_by_x_in_y_eq = y_by_x_in_y_eq
                    print('success.')
                else:
                    y_by_x_in_y_eq = self.y_by_x_in_y_eq

                # subs dict
                subs_codes = [f'{expr.var_name} = {expr.code}'
                              for expr in y_eq_group.sub_exprs[:-1]]
                subs_codes.append(f'{self.y_var} = {y_by_x_in_y_eq}')
                # mapping x -> y
                func_codes = [f'def get_y_by_x({self.x_var}):'] + subs_codes[:-1]
                func_codes.append(f'return {y_by_x_in_y_eq}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
                f_get_y_by_x = eq_y_scope['get_y_by_x']
                can_substitute_y_group_to_x_group = True
            except NotImplementedError:
                print('failed because the equation is too complex.')
            except KeyboardInterrupt:
                print(f'failed because {timeout_len} s timeout.')

        if not can_substitute_y_group_to_x_group and not self.options.escape_sympy_solver:

            # 2. try to solve `f(x, y) = 0` to `x = h(y)`
            try:
                if self.x_by_y_in_y_eq is None:
                    print(f'SymPy solve "f({self.x_var}, {self.y_var}) = 0" to "{self.x_var} = h({self.y_var})", ',
                          end='')
                    f = timeout(timeout_len)(lambda: sympy.solve(y_eq, sympy.Symbol(self.x_var, real=True)))
                    x_by_y_in_y_eq = f()
                    if len(x_by_y_in_y_eq) > 1:
                        raise NotImplementedError('Multiple values.')
                    x_by_y_in_y_eq = sympy_tools.sympy2str(x_by_y_in_y_eq[0])
                    self.x_by_y_in_y_eq = x_by_y_in_y_eq
                    print('success.')
                else:
                    x_by_y_in_y_eq = self.x_by_y_in_y_eq
                # subs dict
                subs_codes = [f'{expr.var_name} = {expr.code}'
                              for expr in y_eq_group.sub_exprs[:-1]]
                subs_codes.append(f'{self.y_var} = {x_by_y_in_y_eq}')
                # mapping y -> x
                func_codes = [f'def get_x_by_y({self.y_var}):'] + subs_codes[:-1]
                func_codes.append(f'return {x_by_y_in_y_eq}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
                f_get_x_by_y = eq_y_scope['get_x_by_y']
                can_substitute_x_group_to_y_group = True
            except NotImplementedError:
                print('failed because the equation is too complex.')
            except KeyboardInterrupt:
                print(f'failed because {timeout_len} s timeout.')

        # solve x_equations #
        # ----------------- #

        if not can_substitute_y_group_to_x_group and not self.options.escape_sympy_solver:

            # 3. try to solve `g(x, y) = 0` to `y = l(x)`
            try:
                if self.x_by_y_in_x_eq is None:
                    print(f'SymPy solve "g({self.x_var}, {self.y_var}) = 0" to "{self.x_var} = l({self.y_var})", ',
                          end='')
                    f = timeout(timeout_len)(lambda: sympy.solve(x_eq, sympy.Symbol(self.x_var, real=True)))
                    x_by_y_in_x_eq = f()
                    if len(x_by_y_in_x_eq) > 1:
                        raise NotImplementedError('Multiple solved values.')
                    x_by_y_in_x_eq = sympy_tools.sympy2str(x_by_y_in_x_eq[0])
                    self.x_by_y_in_x_eq = x_by_y_in_x_eq
                    print('success.')
                else:
                    x_by_y_in_x_eq = self.x_by_y_in_x_eq
                # subs dict
                subs_codes = [f'{expr.var_name} = {expr.code}'
                              for expr in x_eq_group.sub_exprs[:-1]]
                subs_codes.append(f'{self.x_var} = {x_by_y_in_x_eq}')
                # mapping y -> x
                func_codes = [f'def get_x_by_y({self.y_var}):'] + subs_codes[:-1]
                func_codes.append(f'return {x_by_y_in_x_eq}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                f_get_x_by_y = eq_x_scope['get_x_by_y']
                can_substitute_x_group_to_y_group = True
            except NotImplementedError:
                print('failed because the equation is too complex.')
            except KeyboardInterrupt:
                print(f'failed because {timeout_len} s timeout.')

            # 4. try to solve `g(x, y) = 0` to `x = l(y)`
            if not can_substitute_x_group_to_y_group and not self.options.escape_sympy_solver:
                try:
                    if self.y_by_x_in_x_eq is None:
                        print(f'SymPy solve "g({self.x_var}, {self.y_var}) = 0" to "{self.y_var} = l({self.x_var})", ',
                              end='')
                        f = timeout(timeout_len)(lambda: sympy.solve(x_eq, sympy.Symbol(self.y_var, real=True)))
                        y_by_x_in_x_eq = f()
                        if len(y_by_x_in_x_eq) > 1:
                            raise NotImplementedError('Multiple values.')
                        y_by_x_in_x_eq = sympy_tools.sympy2str(y_by_x_in_x_eq[0])
                        self.y_by_x_in_x_eq = y_by_x_in_x_eq
                        print('success.')
                    else:
                        y_by_x_in_x_eq = self.y_by_x_in_x_eq
                    # subs dict
                    subs_codes = [f'{expr.var_name} = {expr.code}'
                                  for expr in x_eq_group.sub_exprs[:-1]]
                    subs_codes.append(f'{self.y_var} = {y_by_x_in_x_eq}')
                    # mapping x -> y
                    func_codes = [f'def get_y_by_x({self.x_var}):'] + subs_codes[:-1]
                    func_codes.append(f'return {y_by_x_in_x_eq}')
                    exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                    f_get_y_by_x = eq_x_scope['get_y_by_x']
                    can_substitute_x_group_to_y_group = True
                except NotImplementedError:
                    print('failed because the equation is too complex.')
                except KeyboardInterrupt:
                    print(f'failed because {timeout_len} s timeout.')

        # get fixed points #
        # ---------------- #

        eq_xy_scope = deepcopy(self.pars_update)
        eq_xy_scope.update(self.fixed_vars)
        eq_xy_scope.update(sympy_tools.get_mapping_scope())
        eq_xy_scope.update(x_eq_group['diff_eq'].func_scope)
        eq_xy_scope.update(y_eq_group['diff_eq'].func_scope)
        for key in eq_xy_scope.keys():
            v = eq_xy_scope[key]
            if callable(v):
                eq_xy_scope[key] = tools.numba_func(v, self.pars_update)

        if can_substitute_y_group_to_x_group:
            if f_get_y_by_x is not None:
                func_codes = [f'def optimizer_x({self.x_var}):'] + subs_codes
                func_codes.extend([f'{expr.var_name} = {expr.code}'
                                   for expr in x_eq_group.sub_exprs[:-1]])
                func_codes.append(f'return {sympy_tools.sympy2str(x_eq)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                optimizer = eq_xy_scope['optimizer_x']
                optimizer = njit(optimizer)
                x_range = np.arange(*self.target_vars[self.x_var], resolution)
                x_values = find_root_of_1d(optimizer, x_range)
                x_values = np.array(x_values)
                y_values = f_get_y_by_x(x_values)

            else:
                func_codes = [f'def optimizer_y({self.y_var}):'] + subs_codes
                func_codes.extend([f'{expr.var_name} = {expr.code}'
                                   for expr in x_eq_group.sub_exprs[:-1]])
                func_codes.append(f'return {sympy_tools.sympy2str(x_eq)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                optimizer = eq_xy_scope['optimizer_y']
                optimizer = njit(optimizer)
                y_range = np.arange(*self.target_vars[self.y_var], resolution)
                y_values = find_root_of_1d(optimizer, y_range)
                y_values = np.array(y_values)
                x_values = f_get_x_by_y(y_values)

        elif can_substitute_x_group_to_y_group:
            if f_get_y_by_x is not None:
                func_codes = [f'def optimizer_x({self.x_var}):'] + subs_codes
                func_codes.extend([f'{expr.var_name} = {expr.code}'
                                   for expr in y_eq_group.sub_exprs[:-1]])
                func_codes.append(f'return {sympy_tools.sympy2str(y_eq)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                optimizer = eq_xy_scope['optimizer_x']
                optimizer = njit(optimizer)
                x_range = np.arange(*self.target_vars[self.x_var], resolution)
                x_values = find_root_of_1d(optimizer, x_range)
                x_values = np.array(x_values)
                y_values = f_get_y_by_x(x_values)

            else:
                func_codes = [f'def optimizer_y({self.y_var}):'] + subs_codes
                func_codes.extend([f'{expr.var_name} = {expr.code}'
                                   for expr in y_eq_group.sub_exprs[:-1]])
                func_codes.append(f'return {sympy_tools.sympy2str(y_eq)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                optimizer = eq_xy_scope['optimizer_y']
                optimizer = njit(optimizer)
                y_range = np.arange(*self.target_vars[self.y_var], resolution)
                y_values = find_root_of_1d(optimizer, y_range)
                y_values = np.array(y_values)
                x_values = f_get_x_by_y(y_values)

        else:
            # f
            func_codes = [f'def f_x({self.x_var}, {self.y_var}):']
            func_codes.extend([f'{expr.var_name} = {expr.code}'
                               for expr in x_eq_group.old_exprs[:-1]])
            func_codes.append(f'return {x_eq_group.old_exprs[-1].code}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
            f_x = eq_x_scope['f_x']
            # g
            func_codes = [f'def g_y({self.x_var}, {self.y_var}):']
            func_codes.extend([f'{expr.var_name} = {expr.code}'
                               for expr in y_eq_group.old_exprs[:-1]])
            func_codes.append(f'return {y_eq_group.old_exprs[-1].code}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
            g_y = eq_y_scope['g_y']
            # f**2 + g**2
            optimizer = lambda x: f_x(x[0], x[1]) ** 2 + g_y(x[0], x[1]) ** 2
            # optimization results
            points = find_root_of_2d(optimizer,
                                     x_bound=self.target_vars[self.x_var],
                                     y_bound=self.target_vars[self.y_var],
                                     shgo_args=self.options.shgo_args,
                                     fl_tol=self.options.fl_tol,
                                     xl_tol=self.options.xl_tol)
            x_values, y_values = [], []
            for p in points:
                x_values.append(p[0])
                y_values.append(p[1])

        # jacobian matrix
        f_jacobian = self.get_f_jacobian()

        # stability analysis #
        # ------------------ #

        container = {a: {'x': [], 'y': []} for a in get_2d_classification()}

        for i in range(len(x_values)):
            x = x_values[i]
            y = y_values[i]

            jacobian = f_jacobian(x, y)
            fp_type = stability_analysis(jacobian)

            print(f"Fixed point #{i + 1} at {self.x_var}={x}, {self.y_var}={y} is a {fp_type}.")
            container[fp_type]['x'].append(x)
            container[fp_type]['y'].append(y)

        # visualization #
        # ------------- #

        for fp_type, points in container.items():
            if len(points['x']):
                plot_style = plot_scheme[fp_type]
                plt.plot(points['x'], points['y'], '.', markersize=20, **plot_style, label=fp_type)

        plt.legend()

        if show:
            plt.show()

        return x_values, y_values

    def plot_nullcline(self, resolution=0.1, show=False):
        y_eq = sympy_tools.str2sympy(self.target_eqs[self.y_var].sub_exprs[-1].code)
        x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].sub_exprs[-1].code)
        y_group = self.target_eqs[self.y_var]
        x_group = self.target_eqs[self.x_var]
        x_range = np.arange(*self.target_vars[self.x_var], resolution)
        y_range = np.arange(*self.target_vars[self.y_var], resolution)
        x_style = dict(color='lightcoral', alpha=.7, )
        y_style = dict(color='cornflowerblue', alpha=.7, )

        timeout_len = self.options.sympy_solver_timeout

        # Nullcline of the y variable
        eq_y_scope = deepcopy(self.pars_update)
        eq_y_scope.update(self.fixed_vars)
        eq_y_scope.update(sympy_tools.get_mapping_scope())
        eq_y_scope.update(y_group.diff_eq.func_scope)

        sympy_failed = True
        # 1. try to solve `f(x, y) = 0` to `y = h(x)`
        if not self.options.escape_sympy_solver:
            try:
                if self.y_by_x_in_y_eq is None:
                    print(f'SymPy solve "f({self.x_var}, {self.y_var}) = 0" to '
                          f'"{self.y_var} = h({self.x_var})", ', end='')
                    f = timeout(timeout_len)(lambda: sympy.solve(y_eq, sympy.Symbol(self.y_var, real=True)))
                    y_by_x_in_y_eq = f()
                    if len(y_by_x_in_y_eq) > 1:
                        raise NotImplementedError('Do not support multiple values.')
                    y_by_x_in_y_eq = sympy_tools.sympy2str(y_by_x_in_y_eq[0])
                    self.y_by_x_in_y_eq = y_by_x_in_y_eq
                    print('success.')
                else:
                    y_by_x_in_y_eq = self.y_by_x_in_y_eq
                sympy_failed = False
                func_code = f'def func({self.x_var}):\n'
                for expr in y_group.sub_exprs[:-1]:
                    func_code += f'  {expr.var_name} = {expr.code}\n'
                func_code += f'  return {y_by_x_in_y_eq}'
                exec(compile(func_code, '', 'exec'), eq_y_scope)
                func = eq_y_scope['func']
                try:
                    y_val = func(x_range)
                except TypeError:
                    raise ModelUseError('Missing variables. Please check and set missing '
                                        'variables to "fixed_vars".')
                plt.plot(x_range, y_val, **y_style, label=f"{self.y_var} nullcline")
            except NotImplementedError:
                print('failed because the equation is too complex.')
                sympy_failed = True
            except KeyboardInterrupt:
                print(f'failed because {timeout_len} s timeout.')
                sympy_failed = True

        # 2. try to solve `f(x, y) = 0` to `x = h(y)`
        if sympy_failed and not self.options.escape_sympy_solver:
            try:
                if self.x_by_y_in_y_eq is None:
                    print(f'SymPy solve "f({self.x_var}, {self.y_var}) = 0" to '
                          f'"{self.x_var} = h({self.y_var})", ', end='')
                    f = timeout(timeout_len)(lambda: sympy.solve(y_eq, sympy.Symbol(self.x_var, real=True)))
                    x_by_y_in_y_eq = f()
                    if len(x_by_y_in_y_eq) > 1:
                        raise NotImplementedError('Multiple values.')
                    x_by_y_in_y_eq = sympy_tools.sympy2str(x_by_y_in_y_eq[0])
                    self.x_by_y_in_y_eq = x_by_y_in_y_eq
                else:
                    x_by_y_in_y_eq = self.x_by_y_in_y_eq
                sympy_failed = False
                func_code = f'def func({self.y_var}):\n'
                for expr in y_group.sub_exprs[:-1]:
                    func_code += f'  {expr.var_name} = {expr.code}\n'
                func_code += f'  return {x_by_y_in_y_eq}'
                exec(compile(func_code, '', 'exec'), eq_y_scope)
                func = eq_y_scope['func']
                try:
                    x_val = func(y_range)
                except TypeError:
                    raise ModelUseError('Missing variables. Please check and set missing '
                                        'variables to "fixed_vars".')
                plt.plot(x_val, y_range, **y_style, label=f"{self.y_var} nullcline")
            except NotImplementedError:
                print('failed because the equation is too complex.')
                sympy_failed = True
            except KeyboardInterrupt:
                print(f'failed because {timeout_len} s timeout.')
                sympy_failed = True

        # use optimization method
        if sympy_failed:
            func_codes = [f'def optimizer_x({self.x_var}, {self.y_var}):']
            for expr in y_group.old_exprs[:-1]:
                func_codes.append(f'{expr.var_name} = {expr.code}')
            func_codes.append(f'return {y_group.old_exprs[-1].code}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
            optimizer = eq_y_scope['optimizer_x']
            optimizer = njit(optimizer)

            x_values, y_values = [], []
            for y in y_range:
                xs = find_root_of_1d(optimizer, x_range, args=(y,))
                for x in xs:
                    x_values.append(x)
                    y_values.append(y)
            x_values = np.array(x_values)
            y_values = np.array(y_values)

            plt.plot(x_values, y_values, '.', **y_style,
                     label=f"{self.y_var} nullcline")

        # Nullcline of the x variable
        eq_x_scope = deepcopy(self.pars_update)
        eq_x_scope.update(self.fixed_vars)
        eq_x_scope.update(sympy_tools.get_mapping_scope())
        eq_x_scope.update(x_group.diff_eq.func_scope)

        sympy_failed = True

        # 3. try to solve `g(x, y) = 0` to `y = l(x)`
        if not self.options.escape_sympy_solver:
            try:
                if self.y_by_x_in_x_eq is None:
                    print(f'SymPy solve "g({self.x_var}, {self.y_var}) = 0" to "{self.y_var} = l({self.x_var})", ',
                          end='')
                    f = timeout(timeout_len)(lambda: sympy.solve(x_eq, sympy.Symbol(self.y_var, real=True)))
                    y_by_x_in_x_eq = f()
                    if len(y_by_x_in_x_eq) > 1:
                        raise NotImplementedError('Multiple values.')
                    y_by_x_in_x_eq = sympy_tools.sympy2str(y_by_x_in_x_eq[0])
                    self.y_by_x_in_x_eq = y_by_x_in_x_eq
                    print('success.')
                else:
                    y_by_x_in_x_eq = self.y_by_x_in_x_eq
                sympy_failed = False
                func_code = f'def func({self.x_var}):\n'
                for expr in x_group.sub_exprs[:-1]:
                    func_code += f'  {expr.var_name} = {expr.code}\n'
                func_code += f'  return {y_by_x_in_x_eq}'
                exec(compile(func_code, '', 'exec'), eq_x_scope)
                func = eq_x_scope['func']
                try:
                    y_val = func(x_range)
                except TypeError:
                    raise ModelUseError('Missing variables. Please check and set missing '
                                        'variables to "fixed_vars".')
                plt.plot(x_range, y_val, **x_style, label=f"{self.x_var} nullcline")
            except NotImplementedError:
                sympy_failed = True
                print('failed because the equation is too complex.')
            except KeyboardInterrupt:
                sympy_failed = True
                print(f'failed because {timeout_len} s timeout.')

        # 4. try to solve `g(x, y) = 0` to `x = l(y)`
        if sympy_failed and not self.options.escape_sympy_solver:
            try:
                if self.x_by_y_in_x_eq is None:
                    print(f'SymPy solve "g({self.x_var}, {self.y_var}) = 0" to "{self.x_var} = l({self.y_var})", ',
                          end='')
                    f = timeout(timeout_len)(lambda: sympy.solve(x_eq, sympy.Symbol(self.x_var, real=True)))
                    x_by_y_in_x_eq = f()
                    if len(x_by_y_in_x_eq) > 1:
                        raise NotImplementedError('Multiple solved values.')
                    x_by_y_in_x_eq = sympy_tools.sympy2str(x_by_y_in_x_eq[0])
                    self.x_by_y_in_x_eq = x_by_y_in_x_eq
                    print('success.')
                else:
                    x_by_y_in_x_eq = self.x_by_y_in_x_eq
                sympy_failed = False
                func_code = f'def func({self.y_var}):\n'
                for expr in x_group.sub_exprs[:-1]:
                    func_code += f'  {expr.var_name} = {expr.code}\n'
                func_code += f'  return {x_by_y_in_x_eq}'
                exec(compile(func_code, '', 'exec'), eq_x_scope)
                func = eq_x_scope['func']
                try:
                    x_val = func(y_range)
                except TypeError:
                    raise ModelUseError('Missing variables. Please check and set missing '
                                        'variables to "fixed_vars".')
                plt.plot(x_val, y_range, **x_style, label=f"{self.x_var} nullcline")
            except NotImplementedError:
                print('failed because the equation is too complex.')
                sympy_failed = True
            except KeyboardInterrupt:
                print(f'failed because {timeout_len} s timeout.')
                sympy_failed = True

        # use optimization method
        if sympy_failed:
            func_codes = [f'def optimizer_x({self.x_var}, {self.y_var}):']
            for expr in x_group.old_exprs[:-1]:
                func_codes.append(f'{expr.var_name} = {expr.code}')
            func_codes.append(f'return {x_group.old_exprs[-1].code}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
            optimizer = eq_x_scope['optimizer_x']
            optimizer = njit(optimizer)

            x_values, y_values = [], []
            for y in y_range:
                xs = find_root_of_1d(optimizer, x_range, (y,))
                for x in xs:
                    x_values.append(x)
                    y_values.append(y)
            x_values = np.array(x_values)
            y_values = np.array(y_values)
            plt.plot(x_values, y_values, '.', **x_style,
                     label=f"{self.x_var} nullcline")

        # finally
        plt.xlabel(self.x_var)
        plt.ylabel(self.y_var)
        scale = (self.options.lim_scale - 1.) / 2
        plt.xlim(*rescale(self.target_vars[self.x_var], scale=scale))
        plt.ylim(*rescale(self.target_vars[self.y_var], scale=scale))
        plt.legend()
        if show:
            plt.show()

    def plot_trajectory(self, target_setting, axes='v-v', inputs=(), show=False):
        """Plot trajectories according to the settings.

        When target_vars = ['m', 'n']
        then, target_setting can be: (initial v1, initial v2, duration)
                          (0., 1., 100.)       # initial values: m=0., n=1., duration=100.
               or,        (0., 1., (10., 90.)) # initial values: m=0., n=1., simulation in [10., 90.]
               or,        [(0., 1., (10., 90.)),
                           (0.5, 1.5, 100.)]  # two trajectory

        Parameters
        ----------
        target_setting : list, tuple
            The initial value setting of the targets.
        axes : str
            The axes.
        show : bool
            Whether show or not.
        """
        trajectories = get_trajectories(model=self.model,
                                        target_vars=list(self.target_vars.keys()),
                                        target_setting=target_setting,
                                        pars_update=self.pars_update,
                                        fixed_vars=self.fixed_vars,
                                        inputs=inputs)
        if axes == 'v-v':
            for trajectory in trajectories:
                plt.plot(getattr(trajectory, self.x_var),
                         getattr(trajectory, self.y_var),
                         label=trajectory.legend)
            plt.xlabel(self.x_var)
            plt.ylabel(self.y_var)
            scale = (self.options.lim_scale - 1.) / 2
            plt.xlim(*rescale(self.target_vars[self.x_var], scale=scale))
            plt.ylim(*rescale(self.target_vars[self.y_var], scale=scale))
            plt.legend()
        elif axes == 't-v':
            for trajectory in trajectories:
                plt.plot(trajectory.ts,
                         getattr(trajectory, self.x_var),
                         label=trajectory.legend + f', {self.x_var}')
                plt.plot(trajectory.ts,
                         getattr(trajectory, self.y_var),
                         label=trajectory.legend + f', {self.y_var}')
            plt.legend(title='Initial values')
        else:
            raise ModelUseError(f'Unknown axes "{axes}", only support "v-v" and "t-v".')
        if show:
            plt.show()


def get_trajectories(
        model: NeuType,
        target_vars: typing.Union[typing.List[str], typing.Tuple[str]],
        target_setting: typing.Union[typing.List, typing.Tuple],
        fixed_vars: typing.Dict = None,
        inputs: typing.Union[typing.List, typing.Tuple] = (),
        pars_update: typing.Dict = None,
):
    """Get trajectories.

    Parameters
    ----------
    model : NeuType
        The neuron model.
    target_vars : list, tuple
        The target variables.
    target_setting : dict
        The initial value setting of the targets.
    fixed_vars : dict
        The fixed variables.
    inputs : list, tuple
        The model inputs.
    pars_update : dict
        The parameters to update.

    Returns
    -------
    trajectories : list
        The trajectories.
    """
    # check initial values
    # ---------------------
    # When target_vars = ['m', 'n']
    # then, target_setting can be: (initial v1, initial v2, duration)
    #                   (0., 1., 100.)       # initial values: m=0., n=1., duration=100.
    #        or,        (0., 1., (10., 90.)) # initial values: m=0., n=1., simulation in [10., 90.]
    #        or,        [(0., 1., (10., 90.)),
    #                    (0.5, 1.5, 100.)]  # two trajectory

    durations = []
    simulating_duration = [np.inf, -np.inf]

    # format target setting
    if isinstance(target_setting[0], (int, float)):
        target_setting = (target_setting,)

    # initial values
    initials = np.zeros((len(target_vars), len(target_setting)), dtype=np.float_)

    # format duration and initial values
    for i, setting in enumerate(target_setting):
        # checking
        try:
            assert isinstance(setting, (tuple, list))
            assert len(setting) == len(target_vars) + 1
        except AssertionError:
            raise ModelUseError('"target_setting" be a tuple with the format of '
                                '(var1 initial, var2 initial, ..., duration).')
        # duration
        duration = setting[-1]
        if isinstance(duration, (int, float)):
            durations.append([0., duration])
            if simulating_duration[0] > 0.:
                simulating_duration[0] = 0.
            if simulating_duration[1] < duration:
                simulating_duration[1] = duration
        elif isinstance(duration, (tuple, list)):
            try:
                assert len(duration) == 2
                assert duration[0] < duration[1]
            except AssertionError:
                raise ModelUseError('duration specification must be a tuple/list with '
                                    'the form of (start, end).')
            durations.append(list(duration))
            if simulating_duration[0] > duration[0]:
                simulating_duration[0] = duration[0]
            if simulating_duration[1] < duration[1]:
                simulating_duration[1] = duration[1]
        else:
            raise ValueError(f'Unknown duration type "{type(duration)}", {duration}')
        # initial values
        for j, val in enumerate(setting[:-1]):
            initials[j, i] = val

    # initialize neuron group
    num = len(target_setting) if len(target_setting) else 1
    group = NeuGroup(model, geometry=num, monitors=target_vars, pars_update=pars_update)
    for i, key in enumerate(target_vars):
        group.ST[key] = initials[i]

    # initialize runner
    group.runner = TrajectoryRunner(group, target_vars=target_vars, fixed_vars=fixed_vars)

    # run
    group.run(duration=simulating_duration, inputs=inputs)

    # monitors
    trajectories = []
    times = group.mon.ts
    dt = profile.get_dt()
    for i, setting in enumerate(target_setting):
        duration = durations[i]
        start = int((duration[0] - simulating_duration[0]) / dt)
        end = int((duration[1] - simulating_duration[0]) / dt)
        trajectory = tools.DictPlus()
        legend = 'traj, '
        trajectory['ts'] = times[start: end]
        for j, var in enumerate(target_vars):
            legend += f'${var}_0$={setting[j]}, '
            trajectory[var] = getattr(group.mon, var)[start: end, i]
        if legend.strip():
            trajectory['legend'] = legend[:-2]
        trajectories.append(trajectory)
    return trajectories
