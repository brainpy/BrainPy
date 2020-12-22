# -*- coding: utf-8 -*-

import typing
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as onp
import sympy
from mpl_toolkits.mplot3d import Axes3D
from numba import njit

from .solver import find_root
from .utils import get_1d_classification
from .utils import get_2d_classification
from .utils import plot_scheme
from .utils import stability_analysis
from .. import tools
from ..core_system import NeuType
from ..errors import ModelUseError
from ..integration import sympy_tools

__all__ = [
    'BifurcationAnalyzer',
]


class BifurcationAnalyzer(object):
    """A tool class for bifurcation analysis.
    
    The bifurcation analyzer is restricted to analyze the bifurcation
    relation between membrane potential and a given model parameter
    (codimension-1 case) or two model parameters (codimension-2 case).
    
    Externally injected current is also treated as a model parameter in
    this class, instead of a model state.

    Parameters
    ----------

    model :  NeuType
        An abstract neuronal type defined in BrainPy.

    """

    def __init__(self, model, target_pars, dynamical_vars, fixed_vars=None,
                 pars_update=None, par_resolution=0.1, var_resolution=0.1):

        # check "model"
        try:
            assert isinstance(model, NeuType)
        except AssertionError:
            raise ModelUseError('Bifurcation analysis only support neuron type model.')
        self.model = model

        # check "target_pars"
        try:
            assert isinstance(target_pars, dict)
        except AssertionError:
            raise ModelUseError('"target_pars" must a dict with the format of: '
                                '{"Parameter A": [A_min, A_max], "Parameter B": [B_min, B_max]}')
        self.target_pars = target_pars
        if len(target_pars) > 2:
            raise ModelUseError("The number of parameters in bifurcation"
                                "analysis cannot exceed 2.")

        # check "fixed_vars"
        if fixed_vars is None:
            fixed_vars = dict()
        try:
            assert isinstance(fixed_vars, dict)
        except AssertionError:
            raise ModelUseError('"fixed_vars" must be a dict the format of: '
                                '{"Variable A": A_value, "Variable B": B_value}')
        self.fixed_vars = fixed_vars

        # check "dynamical_vars"
        try:
            assert isinstance(dynamical_vars, dict)
        except AssertionError:
            raise ModelUseError('"target_vars" must a dict with the format of: '
                                '{"Variable A": [A_min, A_max], "Variable B": [B_min, B_max]}')
        self.dynamical_vars = dynamical_vars

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

        # bifurcation analysis
        if len(self.dynamical_vars) == 1:
            self.analyzer = _1DSystemAnalyzer(model=model,
                                              target_pars=target_pars,
                                              dynamical_vars=dynamical_vars,
                                              fixed_vars=fixed_vars,
                                              pars_update=pars_update,
                                              par_resolution=par_resolution,
                                              var_resolution=var_resolution)

        elif len(self.dynamical_vars) == 2:
            self.analyzer = _2DSystemAnalyzer(model=model,
                                              target_pars=target_pars,
                                              dynamical_vars=dynamical_vars,
                                              fixed_vars=fixed_vars,
                                              pars_update=pars_update,
                                              par_resolution=par_resolution,
                                              var_resolution=var_resolution)

        else:
            raise ModelUseError(f'Cannot analyze three dimensional system: {self.dynamical_vars}')

    def plot_bifurcation(self, plot_vars=(), show=False):
        if isinstance(plot_vars, str):
            plot_vars = [plot_vars]
        try:
            assert isinstance(plot_vars, (tuple, list))
        except AssertionError:
            raise ModelUseError('"plot_vars" must a tuple/list.')
        for var in plot_vars:
            if var in self.fixed_vars:
                raise ModelUseError(f'"{var}" is defined in "fixed_vars", '
                                    f'cannot be used to plot.')
            if var not in self.dynamical_vars:
                raise ModelUseError(f'"{var}" is not a dynamical variable, '
                                    f'cannot be used to plot.')
        if len(plot_vars) == 0:
            plot_vars = list(self.dynamical_vars.keys())

        self.analyzer.plot_bifurcation(plot_vars=plot_vars, show=show)


class _CoDimAnalyzer(object):
    def __init__(self,
                 model: NeuType,
                 target_pars: typing.Dict,
                 dynamical_vars: typing.Dict,
                 fixed_vars: typing.Dict = None,
                 pars_update: typing.Dict = None,
                 par_resolution: float = 0.1,
                 var_resolution: float = 0.1, ):
        self.model = model
        self.target_pars = OrderedDict(target_pars)
        self.dynamical_vars = dynamical_vars
        self.pars_update = pars_update
        self.par_resolution = par_resolution
        self.var_resolution = var_resolution

        # check "fixed_vars"
        self.fixed_vars = dict()
        for integrator in model.integrators:
            var_name = integrator.diff_eq.var_name
            if var_name not in dynamical_vars:
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
        for key in self.dynamical_vars.keys():
            if key not in var2eq:
                raise ModelUseError(f'target "{key}" is not a dynamical variable.')
            integrator = var2eq[key]
            diff_eq = integrator.diff_eq
            exprs = diff_eq.get_f_expressions(substitute_vars=list(self.dynamical_vars.keys()))
            self.target_eqs[key] = tools.DictPlus(dependent_expr=exprs[-1],
                                                  exprs=exprs,
                                                  diff_eq=diff_eq)

        self.f_fixed_point = None

    def plot_bifurcation(self, vars, show=False):
        raise NotImplementedError


class _1DSystemAnalyzer(_CoDimAnalyzer):
    """Bifurcation analysis of 1D system.

    Using this class, we can make co-dimension1 or co-dimension2 bifurcation analysis.
    """

    def __init__(self, model, target_pars, dynamical_vars, fixed_vars=None,
                 pars_update=None, par_resolution=0.1, var_resolution=0.1):
        super(_1DSystemAnalyzer, self).__init__(model=model,
                                                target_pars=target_pars,
                                                dynamical_vars=dynamical_vars,
                                                fixed_vars=fixed_vars,
                                                pars_update=pars_update,
                                                par_resolution=par_resolution,
                                                var_resolution=var_resolution)
        self.x_var = list(self.dynamical_vars.keys())[0]
        self.f_dx = None
        self.f_dfdx = None

    def get_f_fixed_point(self):
        if self.f_fixed_point is None:
            x_eq = sympy_tools.str2sympy(self.target_eqs[self.x_var].dependent_expr.code)
            x_group = self.target_eqs[self.x_var]

            # function scope
            eq_x_scope = deepcopy(self.pars_update)
            eq_x_scope.update(self.fixed_vars)
            eq_x_scope.update(sympy_tools.get_mapping_scope())
            eq_x_scope.update(x_group.diff_eq.func_scope)

            # optimizer
            arg_of_pars = ', '.join(list(self.target_pars.keys()))
            func_codes = [f'def optimizer_x({self.x_var}, {arg_of_pars}):']
            for expr in x_group['exprs'][:-1]:
                func_codes.append(f'{expr.var_name} = {expr.code}')
            func_codes.append(f'return {sympy_tools.sympy2str(x_eq)}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
            optimizer = eq_x_scope['optimizer_x']
            optimizer = njit(optimizer)

            # function
            x_range = onp.arange(*self.dynamical_vars[self.x_var], self.var_resolution)
            scope = {'optimizer': optimizer, 'find_root': find_root,
                     'onp': onp, 'x_range': x_range}
            func_codes = [f'def solve_x({arg_of_pars}):']
            func_codes.append(f'x_values = find_root(optimizer, x_range, ({arg_of_pars}, ))')
            func_codes.append('return onp.array(x_values)')
            exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
            self.f_fixed_point = scope['solve_x']

        return self.f_fixed_point

    def get_f_dx(self):
        if self.f_dx is None:
            arg_of_pars = ', '.join(list(self.target_pars.keys()))
            eqs_of_x = self.target_eqs[self.x_var]
            scope = deepcopy(self.pars_update)
            scope.update(self.fixed_vars)
            scope.update(sympy_tools.get_mapping_scope())
            scope.update(eqs_of_x.diff_eq.func_scope)
            func_code = f'def func({self.x_var}, {arg_of_pars}):\n'
            for expr in eqs_of_x.exprs[:-1]:
                func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {eqs_of_x.exprs[-1].code}'
            exec(compile(func_code, '', 'exec'), scope)
            func = scope['func']
            self.f_dx = func
        return self.f_dx

    def get_f_derivative(self):
        if self.f_dfdx is None:
            arg_of_pars = ', '.join(list(self.target_pars.keys()))
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
                func_codes = [f'def dfxdx({self.x_var}, {arg_of_pars}):']
                for expr in x_eq_group.exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfxdx_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                dfxdx = eq_x_scope['dfxdx']
            except:
                scope = dict(fx=self.get_f_dx())
                func_codes = [f'def dfxdx({self.x_var}, {arg_of_pars}):']
                func_codes.append(f'origin = fx({self.x_var}, {arg_of_pars}))')
                func_codes.append(f'disturb = fx({self.x_var}+1e-4, {arg_of_pars}))')
                func_codes.append(f'return (disturb - origin) / 1e-4')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfxdx = scope['dfxdx']
            self.f_dfdx = dfxdx
        return self.f_dfdx

    def plot_bifurcation(self, plot_vars=None, show=False):
        f_fixed_point = self.get_f_fixed_point()
        f_dfdx = self.get_f_derivative()

        if len(self.target_pars) == 1:
            container = {c: {'p': [], 'x': []} for c in get_1d_classification()}

            # fixed point
            par_name = list(self.target_pars.keys())[0]
            par_lim = list(self.target_pars.values())[0]
            for p in onp.arange(par_lim[0], par_lim[1], self.par_resolution):
                xs = f_fixed_point(p)
                for x in xs:
                    dfdx = f_dfdx(x, p)
                    fp_type = stability_analysis(dfdx)
                    container[fp_type]['p'].append(p)
                    container[fp_type]['x'].append(x)

            # visualization
            for fp_type, points in container.items():
                if len(points['x']):
                    plot_style = plot_scheme[fp_type]
                    plt.plot(points['p'], points['x'], '.', **plot_style, label=fp_type)
            plt.xlabel(par_name)
            plt.ylabel(self.x_var)
            plt.legend()
            if show:
                plt.show()

        elif len(self.target_pars) == 2:
            container = {c: {'p1': [], 'p2': [], 'x': []}
                         for c in get_1d_classification()}

            # fixed point
            par_names = list(self.target_pars.keys())
            par_lims = list(self.target_pars.values())
            par_lim1 = par_lims[0]
            par_lim2 = par_lims[1]
            for p1 in onp.arange(par_lim1[0], par_lim1[1], self.par_resolution):
                for p2 in onp.arange(par_lim2[0], par_lim2[1], self.par_resolution):
                    xs = f_fixed_point(p1, p2)
                    for x in xs:
                        dfdx = f_dfdx(x, p1, p2)
                        fp_type = stability_analysis(dfdx)
                        container[fp_type]['p1'].append(p1)
                        container[fp_type]['p2'].append(p2)
                        container[fp_type]['x'].append(x)

            # visualization
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for fp_type, points in container.items():
                if len(points['x']):
                    plot_style = plot_scheme[fp_type]
                    xs = points['p1']
                    ys = points['p2']
                    zs = points['x']
                    ax.scatter(xs, ys, zs, **plot_style, label=fp_type)
            ax.set_xlabel(par_names[0])
            ax.set_ylabel(par_names[1])
            ax.set_zlabel(self.x_var)
            ax.grid(True)
            ax.legend()
            if show:
                plt.show()


class _2DSystemAnalyzer(_CoDimAnalyzer):
    def __init__(self, model, target_pars, dynamical_vars, fixed_vars=None,
                 pars_update=None, par_resolution=0.1, var_resolution=0.1):
        super(_2DSystemAnalyzer, self).__init__(model=model,
                                                target_pars=target_pars,
                                                dynamical_vars=dynamical_vars,
                                                fixed_vars=fixed_vars,
                                                pars_update=pars_update,
                                                par_resolution=par_resolution,
                                                var_resolution=var_resolution)
        if isinstance(dynamical_vars, OrderedDict):
            self.x_var, self.y_var = list(dynamical_vars.keys())
        else:
            self.x_var, self.y_var = list(sorted(dynamical_vars.keys()))

        self.f_dx = None
        self.f_dy = None
        self.f_jacobian = None

    def get_f_dx(self):
        if self.f_dx is None:
            eqs_of_x = self.target_eqs[self.x_var]
            scope = deepcopy(self.pars_update)
            scope.update(self.fixed_vars)
            scope.update(sympy_tools.get_mapping_scope())
            scope.update(eqs_of_x.diff_eq.func_scope)
            arg_of_pars = ', '.join(list(self.target_pars.keys()))
            func_code = f'def func({self.x_var}, {self.y_var}, {arg_of_pars}):\n'
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
            arg_of_pars = ', '.join(list(self.target_pars.keys()))
            func_code = f'def func({self.x_var}, {self.y_var}, {arg_of_pars}):\n'
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

            arg_of_pars = ', '.join(list(self.target_pars.keys()))

            # dfxdx
            try:
                dfxdx_expr = sympy.diff(x_eq, x_symbol)
                func_codes = [f'def dfxdx({self.x_var}, {self.y_var}, {arg_of_pars}):']
                for expr in x_eq_group.exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfxdx_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                dfxdx = eq_x_scope['dfxdx']
            except:
                scope = dict(fx=self.get_f_dx())
                func_codes = [f'def dfxdx({self.x_var}, {self.y_var}, {arg_of_pars}):']
                func_codes.append(f'origin = fx({self.x_var}, {self.y_var}, {arg_of_pars}))')
                func_codes.append(f'disturb = fx({self.x_var}+1e-4, {self.y_var}, {arg_of_pars}))')
                func_codes.append(f'return (disturb - origin) / 1e-4')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfxdx = scope['dfxdx']

            # dfxdy
            try:
                dfxdy_expr = sympy.diff(x_eq, y_symbol)
                func_codes = [f'def dfxdy({self.x_var}, {self.y_var}, {arg_of_pars}):']
                for expr in x_eq_group.exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfxdy_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
                dfxdy = eq_x_scope['dfxdy']
            except:
                scope = dict(fx=self.get_f_dx())
                func_codes = [f'def dfxdy({self.x_var}, {self.y_var}, {arg_of_pars}):']
                func_codes.append(f'origin = fx({self.x_var}, {self.y_var}, {arg_of_pars}))')
                func_codes.append(f'disturb = fx({self.x_var}, {self.y_var}+1e-4, {arg_of_pars}))')
                func_codes.append(f'return (disturb - origin) / 1e-4')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfxdy = scope['dfxdy']

            # dfydx
            try:
                dfydx_expr = sympy.diff(y_eq, x_symbol)
                func_codes = [f'def dfydx({self.x_var}, {self.y_var}, {arg_of_pars}):']
                for expr in y_eq_group.exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfydx_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
                dfydx = eq_y_scope['dfydx']
            except:
                scope = dict(fy=self.get_f_dy())
                func_codes = [f'def dfydx({self.x_var}, {self.y_var}, {arg_of_pars}):']
                func_codes.append(f'origin = fy({self.x_var}, {self.y_var}, {arg_of_pars}))')
                func_codes.append(f'disturb = fy({self.x_var}+1e-4, {self.y_var}, {arg_of_pars}))')
                func_codes.append(f'return (disturb - origin) / 1e-4')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfydx = scope['dfydx']

            # dfydy
            try:
                dfydy_expr = sympy.diff(y_eq, y_symbol)
                func_codes = [f'def dfydy({self.x_var}, {self.y_var}, {arg_of_pars}):']
                for expr in y_eq_group.exprs[:-1]:
                    func_codes.append(f'{expr.var_name} = {expr.code}')
                func_codes.append(f'return {sympy_tools.sympy2str(dfydy_expr)}')
                exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
                dfydy = eq_y_scope['dfydy']
            except:
                scope = dict(fy=self.get_f_dy())
                func_codes = [f'def dfydy({self.x_var}, {self.y_var}, {arg_of_pars}):']
                func_codes.append(f'origin = fy({self.x_var}, {self.y_var}, {arg_of_pars}))')
                func_codes.append(f'disturb = fy({self.x_var}, {self.y_var}+1e-4, {arg_of_pars}))')
                func_codes.append(f'return (disturb - origin) / 1e-4')
                exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                dfydy = scope['dfydy']

            # jacobian matrix
            scope = dict(f_dfydy=dfydy, f_dfydx=dfydx,
                         f_dfxdy=dfxdy, f_dfxdx=dfxdx, np=onp)
            func_codes = [f'def f_jacobian({self.x_var}, {self.y_var}, {arg_of_pars}):']
            func_codes.append(f'dfxdx = f_dfxdx({self.x_var}, {self.y_var}, {arg_of_pars})')
            func_codes.append(f'dfxdy = f_dfxdy({self.x_var}, {self.y_var}, {arg_of_pars})')
            func_codes.append(f'dfydx = f_dfydx({self.x_var}, {self.y_var}, {arg_of_pars})')
            func_codes.append(f'dfydy = f_dfydy({self.x_var}, {self.y_var}, {arg_of_pars})')
            func_codes.append('return np.array([[dfxdx, dfxdy], [dfydx, dfydy]])')
            exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
            self.f_jacobian = scope['f_jacobian']

        return self.f_jacobian

    def get_f_fixed_point(self):
        if self.f_fixed_point is None:
            x_eq_group = self.target_eqs[self.x_var]
            y_eq_group = self.target_eqs[self.y_var]
            x_eq = sympy_tools.str2sympy(x_eq_group['dependent_expr'].code)
            y_eq = sympy_tools.str2sympy(y_eq_group['dependent_expr'].code)

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

            arg_of_pars = ', '.join(list(self.target_pars.keys()))

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
                func_codes = [f'def get_y_by_x({self.x_var}, {arg_of_pars}):']
                func_codes += subs_codes[:-1]
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
                    func_codes = [f'def get_x_by_y({self.y_var}, {arg_of_pars}):']
                    func_codes += subs_codes[:-1]
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
                    subs_codes = [f'{expr.var_name} = {expr.code}'
                                  for expr in x_eq_group['exprs'][:-1]]
                    subs_codes.append(f'{self.x_var} = {x_eq_by_y}')
                    # mapping y -> x
                    func_codes = [f'def get_x_by_y({self.y_var}, {arg_of_pars}):']
                    func_codes += subs_codes[:-1]
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
                        func_codes = [f'def get_y_by_x({self.x_var}, {arg_of_pars}):']
                        func_codes += subs_codes[:-1]
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
                    func_codes = [f'def optimizer_x({self.x_var}, {arg_of_pars}):']
                    func_codes += subs_codes
                    func_codes.extend([f'{expr.var_name} = {expr.code}'
                                       for expr in x_eq_group['exprs'][:-1]])
                    func_codes.append(f'return {sympy_tools.sympy2str(x_eq)}')
                    exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                    optimizer = eq_xy_scope['optimizer_x']
                    optimizer = njit(optimizer)

                    x_range = onp.arange(*self.dynamical_vars[self.x_var], self.var_resolution)
                    scope = {'optimizer': optimizer, 'x_range': x_range,
                             'find_root': find_root, 'np': onp,
                             'f_get_y_by_x': f_get_y_by_x}
                    func_codes = [f'def f_fixed_point({arg_of_pars}):']
                    func_codes.append(f'x_values = find_root(optimizer, x_range, ({arg_of_pars},))')
                    func_codes.append(f'x_values = np.array(x_values)')
                    func_codes.append(f'y_values = f_get_y_by_x(x_values, {arg_of_pars})')
                    func_codes.append(f'return x_values, y_values')
                    exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                    f_fixed_point = scope['f_fixed_point']

                else:
                    func_codes = [f'def optimizer_y({self.y_var}, {arg_of_pars}):']
                    func_codes += subs_codes
                    func_codes.extend([f'{expr.var_name} = {expr.code}'
                                       for expr in x_eq_group['exprs'][:-1]])
                    func_codes.append(f'return {sympy_tools.sympy2str(x_eq)}')
                    exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                    optimizer = eq_xy_scope['optimizer_y']
                    optimizer = njit(optimizer)

                    y_range = onp.arange(*self.dynamical_vars[self.y_var], self.var_resolution)
                    scope = {'optimizer': optimizer, 'y_range': y_range,
                             'find_root': find_root, 'np': onp,
                             'f_get_x_by_y': f_get_x_by_y}
                    func_codes = [f'def f_fixed_point({arg_of_pars}):']
                    func_codes.append(f'y_values = find_root(optimizer, y_range, ({arg_of_pars},))')
                    func_codes.append(f'y_values = np.array(y_values)')
                    func_codes.append(f'x_values = f_get_x_by_y(y_values, {arg_of_pars})')
                    func_codes.append(f'return x_values, y_values')
                    exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                    f_fixed_point = scope['f_fixed_point']

            elif can_substitute_x_group_to_y_group:
                if f_get_y_by_x is not None:
                    func_codes = [f'def optimizer_x({self.x_var}, {arg_of_pars}):']
                    func_codes += subs_codes
                    func_codes.extend([f'{expr.var_name} = {expr.code}'
                                       for expr in y_eq_group['exprs'][:-1]])
                    func_codes.append(f'return {sympy_tools.sympy2str(y_eq)}')
                    exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                    optimizer = eq_xy_scope['optimizer_x']
                    optimizer = njit(optimizer)

                    x_range = onp.arange(*self.dynamical_vars[self.x_var], self.var_resolution)
                    scope = {'optimizer': optimizer, 'x_range': x_range,
                             'find_root': find_root, 'np': onp,
                             'f_get_y_by_x': f_get_y_by_x}
                    func_codes = [f'def f_fixed_point({arg_of_pars}):']
                    func_codes.append(f'x_values = find_root(optimizer, x_range, ({arg_of_pars},))')
                    func_codes.append(f'x_values = np.array(x_values)')
                    func_codes.append(f'y_values = f_get_y_by_x(x_values, {arg_of_pars})')
                    func_codes.append(f'return x_values, y_values')
                    exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                    f_fixed_point = scope['f_fixed_point']

                else:
                    func_codes = [f'def optimizer_y({self.y_var}, {arg_of_pars}):']
                    func_codes += subs_codes
                    func_codes.extend([f'{expr.var_name} = {expr.code}'
                                       for expr in y_eq_group['exprs'][:-1]])
                    func_codes.append(f'return {sympy_tools.sympy2str(y_eq)}')
                    exec(compile('\n  '.join(func_codes), '', 'exec'), eq_xy_scope)
                    optimizer = eq_xy_scope['optimizer_y']
                    optimizer = njit(optimizer)

                    y_range = onp.arange(*self.dynamical_vars[self.y_var], self.var_resolution)
                    scope = {'optimizer': optimizer, 'y_range': y_range,
                             'find_root': find_root, 'np': onp,
                             'f_get_x_by_y': f_get_x_by_y}
                    func_codes = [f'def f_fixed_point({arg_of_pars}):']
                    func_codes.append(f'y_values = find_root(optimizer, y_range, ({arg_of_pars},))')
                    func_codes.append(f'y_values = np.array(y_values)')
                    func_codes.append(f'x_values = f_get_x_by_y(y_values, {arg_of_pars})')
                    func_codes.append(f'return x_values, y_values')
                    exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
                    f_fixed_point = scope['f_fixed_point']

            else:
                raise RuntimeError('System Error.')

            self.f_fixed_point = f_fixed_point

        return self.f_fixed_point

    def plot_bifurcation(self, plot_vars, show=False):
        f_fixed_point = self.get_f_fixed_point()
        f_jacobian = self.get_f_jacobian()

        # bifurcation analysis of co-dimension 1
        if len(self.target_pars) == 1:
            container = {c: {'p': [], self.x_var: [], self.y_var: []}
                         for c in get_2d_classification()}

            # fixed point
            par_name = list(self.target_pars.keys())[0]
            par_lim = list(self.target_pars.values())[0]
            for p in onp.arange(par_lim[0], par_lim[1], self.par_resolution):
                xs, ys = f_fixed_point(p)
                for x, y in zip(xs, ys):
                    dfdx = f_jacobian(x, y, p)
                    fp_type = stability_analysis(dfdx)
                    container[fp_type]['p'].append(p)
                    container[fp_type][self.x_var].append(x)
                    container[fp_type][self.y_var].append(y)

            # visualization
            for var in plot_vars:
                plt.figure()
                for fp_type, points in container.items():
                    if len(points['p']):
                        plot_style = plot_scheme[fp_type]
                        plt.plot(points['p'], points[var], '.', **plot_style, label=fp_type)
                plt.xlabel(par_name)
                plt.ylabel(var)
                plt.legend()
            if show:
                plt.show()

        # bifurcation analysis of co-dimension 2
        elif len(self.target_pars) == 2:
            container = {c: {'p1': [], 'p2': [], self.x_var: [], self.y_var: []}
                         for c in get_2d_classification()}

            # fixed point
            par_names = list(self.target_pars.keys())
            par_lims = list(self.target_pars.values())
            par_lim1 = par_lims[0]
            par_lim2 = par_lims[1]
            for p1 in onp.arange(par_lim1[0], par_lim1[1], self.par_resolution):
                for p2 in onp.arange(par_lim2[0], par_lim2[1], self.par_resolution):
                    xs, ys = f_fixed_point(p1, p2)
                    for x, y in zip(xs, ys):
                        dfdx = f_jacobian(x, y, p1, p2)
                        fp_type = stability_analysis(dfdx)
                        container[fp_type]['p1'].append(p1)
                        container[fp_type]['p2'].append(p2)
                        container[fp_type][self.x_var].append(x)
                        container[fp_type][self.y_var].append(y)

            # visualization
            for var in plot_vars:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for fp_type, points in container.items():
                    if len(points['p1']):
                        plot_style = plot_scheme[fp_type]
                        xs = points['p1']
                        ys = points['p2']
                        zs = points[var]
                        ax.scatter(xs, ys, zs, **plot_style, label=fp_type)
                ax.set_xlabel(par_names[0])
                ax.set_ylabel(par_names[1])
                ax.set_zlabel(var)
                ax.legend()
            if show:
                plt.show()


if __name__ == '__main__':
    Axes3D
