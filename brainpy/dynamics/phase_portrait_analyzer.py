# -*- coding: utf-8 -*-

import typing
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import sympy
from numba import njit

from .solver import find_root
from .utils import get_1d_classification
from .utils import get_2d_classification
from .utils import plot_scheme
from .utils import rescale
from .utils import stability_analysis
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
    def __init__(self,
                 model,
                 target_vars,
                 fixed_vars=None,
                 pars_update=None,
                 lim_scale=1.05):
        self.lim_scale = lim_scale

        # check "model"
        try:
            assert isinstance(model, NeuType)
        except AssertionError:
            raise ModelUseError('Phase plane analysis only support neuron type model.')
        self.model = model

        # check "target_vars"
        try:
            assert isinstance(target_vars, dict)
        except AssertionError:
            raise ModelUseError('"target_vars" must a dict with the format of: '
                                '{"Variable A": [A_min, A_max], "Variable B": [B_min, B_max]}')

        self.target_vars = target_vars

        # check "fixed_vars"
        if fixed_vars is None:
            fixed_vars = dict()
        try:
            assert isinstance(fixed_vars, dict)
        except AssertionError:
            raise ModelUseError('"fixed_vars" must be a dict the format of: '
                                '{"Variable A": A_value, "Variable B": B_value}')
        self.fixed_vars = fixed_vars

        # check "pars_update"
        if pars_update is None:
            pars_update = dict()
        try:
            assert isinstance(pars_update, dict)
        except AssertionError:
            raise ModelUseError('"pars_update" must be a dict the format of: '
                                '{"Par A": A_value, "Par B": B_value}')
        for key in pars_update.keys():
            if key not in model.step_scopes:
                raise ModelUseError(f'"{key}" is not a valid parameter in "{model.name}" model. ')
        self.pars_update = pars_update

        # analyzer
        if len(target_vars) == 1:
            self.analyzer = _1DSystemAnalyzer(model=model,
                                              target_vars=target_vars,
                                              fixed_vars=fixed_vars,
                                              pars_update=pars_update,
                                              lim_scale=lim_scale)
        elif len(target_vars) == 2:
            self.analyzer = _2DSystemAnalyzer(model=model,
                                              target_vars=target_vars,
                                              fixed_vars=fixed_vars,
                                              pars_update=pars_update,
                                              lim_scale=lim_scale)
        else:
            raise ModelUseError('BrainPy only support 1D/2D phase plane analysis. '
                                'Or, you can set "fixed_vars" to fix other variables, '
                                'then make 1D/2D phase plane analysis.')

    def plot_vector_field(self, resolution=0.1, lw_lim=(0.5, 5.5), show=False):
        self.analyzer.plot_vector_field(resolution=resolution, lw_lim=lw_lim, show=show)

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
                 lim_scale=1.05):
        self.lim_scale = lim_scale
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
            exprs = diff_eq.get_f_expressions(substitute_vars=list(self.target_vars.keys()))
            self.target_eqs[key] = tools.DictPlus(dependent_expr=exprs[-1],
                                                  exprs=exprs,
                                                  diff_eq=diff_eq)


class _1DSystemAnalyzer(_PPAnalyzer):
    def __init__(self,
                 model,
                 target_vars,
                 fixed_vars=None,
                 pars_update=None,
                 lim_scale=1.05):
        super(_1DSystemAnalyzer, self).__init__(model=model,
                                                target_vars=target_vars,
                                                fixed_vars=fixed_vars,
                                                pars_update=pars_update,
                                                lim_scale=lim_scale)
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
            for expr in eqs_of_x.exprs[:-1]:
                func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {eqs_of_x.exprs[-1].code}'
            exec(compile(func_code, '', 'exec'), scope)
            func = scope['func']
            self.f_dx = func
        return self.f_dx

    def get_f_dfdx(self):
        if self.f_dfdx is None:
            x_symbol = sympy.Symbol(self.x_var, real=True)
            x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].dependent_expr.code)
            x_eq_group = self.target_eqs[self.x_var]

            eq_x_scope = deepcopy(self.pars_update)
            eq_x_scope.update(self.fixed_vars)
            eq_x_scope.update(sympy_tools.get_mapping_scope())
            eq_x_scope.update(x_eq_group['diff_eq'].func_scope)

            # dfxdx
            try:
                dfxdx_expr = sympy.diff(x_eq, x_symbol)
                func_codes = [f'def dfxdx({self.x_var}):']
                for expr in x_eq_group.exprs[:-1]:
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

    def plot_vector_field(self, resolution=0.1, lw_lim=(0.5, 5.5), show=False):
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
        plt.xlim(*rescale(self.target_vars[self.x_var], scale=(self.lim_scale - 1.) / 2))
        plt.legend()
        if show:
            plt.show()

    def plot_fixed_point(self, resolution=0.1, show=False):
        x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].dependent_expr.code)
        x_group = self.target_eqs[self.x_var]

        # function scope
        scope = deepcopy(self.pars_update)
        scope.update(self.fixed_vars)
        scope.update(sympy_tools.get_mapping_scope())
        scope.update(x_group.diff_eq.func_scope)

        try:
            # solve
            results = sympy.solve(x_eq, sympy.Symbol(self.x_var, real=True))

            # function codes
            func_codes = [f'def solve_x():']
            for expr in x_group['exprs'][:-1]:
                func_codes.append(f'{expr.var_name} = {expr.code}')
            return_expr = ', '.join([sympy_tools.sympy2str(expr) for expr in results])
            func_codes.append(f'return {return_expr}')

            # function
            exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
            x_values = scope['solve_x']()
            x_values = np.array([x_values])

        except NotImplementedError:
            # function codes
            func_codes = [f'def optimizer_x({self.x_var}):']
            for expr in x_group['exprs'][:-1]:
                func_codes.append(f'{expr.var_name} = {expr.code}')
            func_codes.append(f'return {sympy_tools.sympy2str(x_eq)}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
            optimizer = scope['optimizer_x']
            optimizer = njit(optimizer)
            x_range = np.arange(*self.target_vars[self.x_var], resolution)
            x_values = find_root(optimizer, x_range)
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
    def __init__(self,
                 model,
                 target_vars,
                 fixed_vars=None,
                 pars_update=None,
                 lim_scale=1.05):
        super(_2DSystemAnalyzer, self).__init__(model=model,
                                                target_vars=target_vars,
                                                fixed_vars=fixed_vars,
                                                pars_update=pars_update,
                                                lim_scale=lim_scale)

        if isinstance(target_vars, OrderedDict):
            self.x_var, self.y_var = list(self.target_vars.keys())
        else:
            self.x_var, self.y_var = list(sorted(self.target_vars.keys()))

        self.f_dy = None
        self.f_dx = None
        self.f_jacobian = None

    def get_f_dx(self):
        if self.f_dx is None:
            eqs_of_x = self.target_eqs[self.x_var]
            scope = deepcopy(self.pars_update)
            scope.update(self.fixed_vars)
            scope.update(sympy_tools.get_mapping_scope())
            scope.update(eqs_of_x.diff_eq.func_scope)
            func_code = f'def func({self.x_var}, {self.y_var}):\n'
            for expr in eqs_of_x.exprs[:-1]:
                func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {eqs_of_x.exprs[-1].code}'
            exec(compile(func_code, '', 'exec'), scope)
            func = scope['func']
            self.f_dx = func
        return self.f_dx

    def get_f_dy(self):
        if self.f_dy is None:
            eqs_of_y = self.target_eqs[self.y_var]
            scope = deepcopy(self.pars_update)
            scope.update(self.fixed_vars)
            scope.update(sympy_tools.get_mapping_scope())
            scope.update(eqs_of_y.diff_eq.func_scope)
            func_code = f'def func({self.x_var}, {self.y_var}):\n'
            for expr in eqs_of_y.exprs[:-1]:
                func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {eqs_of_y.exprs[-1].code}'
            exec(compile(func_code, '', 'exec'), scope)
            self.f_dy = scope['func']
        return self.f_dy

    def get_f_jacobian(self):
        if self.f_jacobian is None:
            x_symbol = sympy.Symbol(self.x_var, real=True)
            y_symbol = sympy.Symbol(self.y_var, real=True)
            x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].dependent_expr.code)
            y_eq = sympy_tools.str2sympy(self.target_eqs[self.y_var].dependent_expr.code)
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
                dfxdx_expr = sympy.diff(x_eq, x_symbol)
                func_codes = [f'def dfxdx({self.x_var}, {self.y_var}):']
                for expr in x_eq_group.exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfxdx_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                dfxdx = eq_x_scope['dfxdx']
            except:
                scope = dict(fx=self.get_f_dx())
                func_codes = [f'def dfxdx({self.x_var}, {self.y_var}):']
                func_codes.append(f'origin = fx({self.x_var}, {self.y_var}))')
                func_codes.append(f'disturb = fx({self.x_var}+1e-4, {self.y_var}))')
                func_codes.append(f'return (disturb - origin) / 1e-4')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfxdx = scope['dfxdx']

            # dfxdy
            try:
                dfxdy_expr = sympy.diff(x_eq, y_symbol)
                func_codes = [f'def dfxdy({self.x_var}, {self.y_var}):']
                for expr in x_eq_group.exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfxdy_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                dfxdy = eq_x_scope['dfxdy']
            except:
                scope = dict(fx=self.get_f_dx())
                func_codes = [f'def dfxdy({self.x_var}, {self.y_var}):']
                func_codes.append(f'origin = fx({self.x_var}, {self.y_var}))')
                func_codes.append(f'disturb = fx({self.x_var}, {self.y_var}+1e-4))')
                func_codes.append(f'return (disturb - origin) / 1e-4')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfxdy = scope['dfxdy']

            # dfydx
            try:
                dfydx_expr = sympy.diff(y_eq, x_symbol)
                func_codes = [f'def dfydx({self.x_var}, {self.y_var}):']
                for expr in y_eq_group.exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfydx_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
                dfydx = eq_y_scope['dfydx']
            except:
                scope = dict(fy=self.get_f_dy())
                func_codes = [f'def dfydx({self.x_var}, {self.y_var}):']
                func_codes.append(f'origin = fy({self.x_var}, {self.y_var}))')
                func_codes.append(f'disturb = fy({self.x_var}+1e-4, {self.y_var}))')
                func_codes.append(f'return (disturb - origin) / 1e-4')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfydx = scope['dfydx']

            # dfydy
            try:
                dfydy_expr = sympy.diff(y_eq, y_symbol)
                func_codes = [f'def dfydy({self.x_var}, {self.y_var}):']
                for expr in y_eq_group.exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfydy_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
                dfydy = eq_y_scope['dfydy']
            except:
                scope = dict(fy=self.get_f_dy())
                func_codes = [f'def dfydy({self.x_var}, {self.y_var}):']
                func_codes.append(f'origin = fy({self.x_var}, {self.y_var}))')
                func_codes.append(f'disturb = fy({self.x_var}, {self.y_var}+1e-4))')
                func_codes.append(f'return (disturb - origin) / 1e-4')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfydy = scope['dfydy']

            # jacobian matrix
            scope = dict(f_dfydy=dfydy, f_dfydx=dfydx,
                         f_dfxdy=dfxdy, f_dfxdx=dfxdx, np=np)
            func_codes = [f'def f_jacobian({self.x_var}, {self.y_var}):']
            func_codes.append(f'dfxdx = f_dfxdx({self.x_var}, {self.y_var})')
            func_codes.append(f'dfxdy = f_dfxdy({self.x_var}, {self.y_var})')
            func_codes.append(f'dfydx = f_dfydx({self.x_var}, {self.y_var})')
            func_codes.append(f'dfydy = f_dfydy({self.x_var}, {self.y_var})')
            func_codes.append('return np.array([[dfxdx, dfxdy], [dfydx, dfydy]])')
            exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
            self.f_jacobian = scope['f_jacobian']

        return self.f_jacobian

    def plot_vector_field(self, resolution=0.1, lw_lim=(0.5, 5.5), show=False):
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
        lw_min, lw_max = lw_lim
        lw = lw_min + lw_max * speed / speed.max()
        plt.streamplot(X, Y, dx, dy, linewidth=lw, arrowsize=1.2, density=1, color='thistle')
        plt.xlabel(self.x_var)
        plt.ylabel(self.y_var)

        if show:
            plt.show()

    def plot_fixed_point(self, resolution=0.1, show=False):
        x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].dependent_expr.code)
        y_eq = sympy_tools.str2sympy(self.target_eqs[self.y_var].dependent_expr.code)
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

        # solve y_equations #
        # ------------------

        try:
            y_eq_by_x = sympy.solve(y_eq, sympy.Symbol(self.y_var, real=True))
            if len(y_eq_by_x) > 1:
                raise ValueError('Multiple values.')
            # subs dict
            y_eq_by_x = sympy_tools.sympy2str(y_eq_by_x[0])
            subs_codes = [f'{expr.var_name} = {expr.code}' for expr in y_eq_group['exprs'][:-1]]
            subs_codes.append(f'{self.y_var} = {y_eq_by_x}')
            # mapping x -> y
            func_codes = [f'def get_y_by_x({self.x_var}):'] + subs_codes[:-1]
            func_codes.append(f'return {y_eq_by_x}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
            f_get_y_by_x = eq_y_scope['get_y_by_x']
            can_substitute_y_group_to_x_group = True

        except NotImplementedError:
            try:
                y_eq_by_y = sympy.solve(y_eq, sympy.Symbol(self.x_var, real=True))
                if len(y_eq_by_y) > 1:
                    raise ValueError('Multiple values.')
                # subs dict
                y_eq_by_y = sympy_tools.sympy2str(y_eq_by_y[0])
                subs_codes = [f'{expr.var_name} = {expr.code}' for expr in y_eq_group['exprs'][:-1]]
                subs_codes.append(f'{self.y_var} = {y_eq_by_y}')
                # mapping y -> x
                func_codes = [f'def get_x_by_y({self.y_var}):'] + subs_codes[:-1]
                func_codes.append(f'return {y_eq_by_y}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
                f_get_x_by_y = eq_y_scope['get_x_by_y']
                can_substitute_x_group_to_y_group = True

            except NotImplementedError:
                pass

        # solve x_equations #
        # ----------------- #

        if not can_substitute_y_group_to_x_group:
            try:
                x_eq_by_y = sympy.solve(x_eq, sympy.Symbol(self.x_var, real=True))
                if len(x_eq_by_y) > 1:
                    raise ValueError('Multiple values.')
                # subs dict
                x_eq_by_y = sympy_tools.sympy2str(x_eq_by_y[0])
                subs_codes = [f'{expr.var_name} = {expr.code}' for expr in x_eq_group['exprs'][:-1]]
                subs_codes.append(f'{self.x_var} = {x_eq_by_y}')
                # mapping y -> x
                func_codes = [f'def get_x_by_y({self.y_var}):'] + subs_codes[:-1]
                func_codes.append(f'return {x_eq_by_y}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                f_get_x_by_y = eq_x_scope['get_x_by_y']
                can_substitute_x_group_to_y_group = True

            except NotImplementedError:
                try:
                    x_eq_by_x = sympy.solve(x_eq, sympy.Symbol(self.y_var, real=True))
                    if len(x_eq_by_x) > 1:
                        raise ValueError('Multiple values.')
                    # subs dict
                    x_eq_by_x = sympy_tools.sympy2str(x_eq_by_x[0])
                    subs_codes = [f'{expr.var_name} = {expr.code}' for expr in x_eq_group['exprs'][:-1]]
                    subs_codes.append(f'{self.y_var} = {x_eq_by_x}')
                    # mapping x -> y
                    func_codes = [f'def get_y_by_x({self.x_var}):'] + subs_codes[:-1]
                    func_codes.append(f'return {x_eq_by_x}')
                    exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                    f_get_y_by_x = eq_x_scope['get_y_by_x']
                    can_substitute_x_group_to_y_group = True

                except NotImplementedError:
                    can_substitute_x_group_to_y_group = False
        else:
            pass

        # checking

        if (not can_substitute_y_group_to_x_group) and (not can_substitute_x_group_to_y_group):
            raise NotImplementedError(f'This model is too complex, we cannot solve '
                                      f'"{self.x_var}" and "{self.y_var}". ')

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
                func_codes.extend([f'{expr.var_name} = {expr.code}' for expr in x_eq_group['exprs'][:-1]])
                func_codes.append(f'return {sympy_tools.sympy2str(x_eq)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                optimizer = eq_xy_scope['optimizer_x']
                optimizer = njit(optimizer)
                x_range = np.arange(*self.target_vars[self.x_var], resolution)
                x_values = find_root(optimizer, x_range)
                x_values = np.array(x_values)
                y_values = f_get_y_by_x(x_values)

            else:
                func_codes = [f'def optimizer_y({self.y_var}):'] + subs_codes
                func_codes.extend([f'{expr.var_name} = {expr.code}' for expr in x_eq_group['exprs'][:-1]])
                func_codes.append(f'return {sympy_tools.sympy2str(x_eq)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                optimizer = eq_xy_scope['optimizer_y']
                optimizer = njit(optimizer)
                y_range = np.arange(*self.target_vars[self.y_var], resolution)
                y_values = find_root(optimizer, y_range)
                y_values = np.array(y_values)
                x_values = f_get_x_by_y(y_values)

        elif can_substitute_x_group_to_y_group:
            if f_get_y_by_x is not None:
                func_codes = [f'def optimizer_x({self.x_var}):'] + subs_codes
                func_codes.extend([f'{expr.var_name} = {expr.code}' for expr in y_eq_group['exprs'][:-1]])
                func_codes.append(f'return {sympy_tools.sympy2str(y_eq)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                optimizer = eq_xy_scope['optimizer_x']
                optimizer = njit(optimizer)
                x_range = np.arange(*self.target_vars[self.x_var], resolution)
                x_values = find_root(optimizer, x_range)
                x_values = np.array(x_values)
                y_values = f_get_y_by_x(x_values)

            else:
                func_codes = [f'def optimizer_y({self.y_var}):'] + subs_codes
                func_codes.extend([f'{expr.var_name} = {expr.code}' for expr in y_eq_group['exprs'][:-1]])
                func_codes.append(f'return {sympy_tools.sympy2str(y_eq)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                optimizer = eq_xy_scope['optimizer_y']
                optimizer = njit(optimizer)
                y_range = np.arange(*self.target_vars[self.y_var], resolution)
                y_values = find_root(optimizer, y_range)
                y_values = np.array(y_values)
                x_values = f_get_x_by_y(y_values)

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
        y_eq = sympy_tools.str2sympy(self.target_eqs[self.y_var].dependent_expr.code)
        x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].dependent_expr.code)
        y_group = self.target_eqs[self.y_var]
        x_group = self.target_eqs[self.x_var]
        x_range = np.arange(*self.target_vars[self.x_var], resolution)
        y_range = np.arange(*self.target_vars[self.y_var], resolution)
        x_style = dict(color='lightcoral', alpha=.7, linewidth=4)
        y_style = dict(color='cornflowerblue', alpha=.7, linewidth=4)

        # Nullcline of the y variable
        eq_y_scope = deepcopy(self.pars_update)
        eq_y_scope.update(self.fixed_vars)
        eq_y_scope.update(sympy_tools.get_mapping_scope())
        eq_y_scope.update(y_group.diff_eq.func_scope)

        try:
            y_eq_by_x = sympy.solve(y_eq, sympy.Symbol(self.y_var, real=True))

            for i, res in enumerate(y_eq_by_x):
                func_code = f'def func({self.x_var}):\n'
                for expr in y_group.exprs[:-1]:
                    func_code += f'  {expr.var_name} = {expr.code}\n'
                func_code += f'  return {sympy_tools.sympy2str(res)}'
                exec(compile(func_code, '', 'exec'), eq_y_scope)
                func = eq_y_scope['func']
                try:
                    y_val = func(x_range)
                except TypeError:
                    raise ModelUseError('Missing variables. Please check and set missing '
                                        'variables to "fixed_vars".')

                label = f"{self.y_var} nullcline" if i == 0 else None
                plt.plot(x_range, y_val, **y_style, label=label)

        except NotImplementedError:
            try:
                y_eq_by_y = sympy.solve(y_eq, sympy.Symbol(self.x_var, real=True))

                for i, res in enumerate(y_eq_by_y):
                    func_code = f'def func({self.y_var}):\n'
                    for expr in y_group.exprs[:-1]:
                        func_code += f'  {expr.var_name} = {expr.code}\n'
                    func_code += f'  return {sympy_tools.sympy2str(res)}'
                    exec(compile(func_code, '', 'exec'), eq_y_scope)
                    func = eq_y_scope['func']
                    try:
                        x_val = func(y_range)
                    except TypeError:
                        raise ModelUseError('Missing variables. Please check and set missing '
                                            'variables to "fixed_vars".')

                    label = f"{self.y_var} nullcline" if i == 0 else None
                    plt.plot(x_val, y_range, **y_style, label=label)
            except NotImplementedError:
                func_codes = [f'def optimizer_x({self.x_var}, {self.y_var}):']
                for expr in y_group['exprs'][:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(y_eq)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
                optimizer = eq_y_scope['optimizer_x']
                optimizer = njit(optimizer)

                x_values, y_values = [], []
                for y in y_range:
                    xs = find_root(optimizer, x_range, (y,))
                    for x in xs:
                        x_values.append(x)
                        y_values.append(y)
                x_values = np.array(x_values)
                y_values = np.array(y_values)

                plt.plot(x_values, y_values, **y_style, label=f"{self.y_var} nullcline")

        # Nullcline of the x variable
        eq_x_scope = deepcopy(self.pars_update)
        eq_x_scope.update(self.fixed_vars)
        eq_x_scope.update(sympy_tools.get_mapping_scope())
        eq_x_scope.update(x_group.diff_eq.func_scope)
        try:
            x_eq_by_x = sympy.solve(x_eq, sympy.Symbol(self.y_var, real=True))

            for i, res in enumerate(x_eq_by_x):
                func_code = f'def func({self.x_var}):\n'
                for expr in x_group.exprs[:-1]:
                    func_code += f'  {expr.var_name} = {expr.code}\n'
                func_code += f'  return {sympy_tools.sympy2str(res)}'
                exec(compile(func_code, '', 'exec'), eq_x_scope)
                func = eq_x_scope['func']
                try:
                    y_val = func(x_range)
                except TypeError:
                    raise ModelUseError('Missing variables. Please check and set missing '
                                        'variables to "fixed_vars".')

                label = f"{self.x_var} nullcline" if i == 0 else None
                plt.plot(x_range, y_val, **x_style, label=label)

        except NotImplementedError:
            try:
                x_eq_by_y = sympy.solve(x_eq, sympy.Symbol(self.x_var, real=True))

                for i, res in enumerate(x_eq_by_y):
                    func_code = f'def func({self.y_var}):\n'
                    for expr in x_group.exprs[:-1]:
                        func_code += f'  {expr.var_name} = {expr.code}\n'
                    func_code += f'  return {sympy_tools.sympy2str(res)}'
                    exec(compile(func_code, '', 'exec'), eq_x_scope)
                    func = eq_x_scope['func']
                    try:
                        x_val = func(y_range)
                    except TypeError:
                        raise ModelUseError('Missing variables. Please check and set missing '
                                            'variables to "fixed_vars".')

                    label = f"{self.x_var} nullcline" if i == 0 else None
                    plt.plot(x_val, y_range, **x_style, label=label)
            except NotImplementedError:
                func_codes = [f'def optimizer_x({self.x_var}, {self.y_var}):']
                for expr in x_group['exprs'][:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(y_eq)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                optimizer = eq_x_scope['optimizer_x']
                optimizer = njit(optimizer)

                x_values, y_values = [], []
                for y in y_range:
                    xs = find_root(optimizer, x_range, (y,))
                    for x in xs:
                        x_values.append(x)
                        y_values.append(y)
                x_values = np.array(x_values)
                y_values = np.array(y_values)
                plt.plot(x_values, y_values, **x_style, label=f"{self.x_var} nullcline")

        # finally
        plt.xlabel(self.x_var)
        plt.ylabel(self.y_var)
        scale = (self.lim_scale - 1.) / 2
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
            scale = (self.lim_scale - 1.) / 2
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
