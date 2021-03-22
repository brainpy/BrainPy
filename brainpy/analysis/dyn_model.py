# -*- coding: utf-8 -*-


import inspect

from brainpy.integrators import ast_analysis
from brainpy.integrators import sympy_analysis

try:
    from numba.core.dispatcher import Dispatcher
except ModuleNotFoundError:
    Dispatcher = None

__all__ = [
    'transform_integrals_to_analyzers',
    'DynamicModel',
]


def transform_integrals_to_analyzers(integrals):
    if callable(integrals):
        integrals = [integrals]

    all_scope = dict()
    all_variables = set()
    all_parameters = set()
    analyzers = []
    for integral in integrals:
        # integral function
        if Dispatcher is not None and isinstance(integral, Dispatcher):
            integral = integral.py_func
        else:
            integral = integral

        # original function
        f = integral.origin_f
        if Dispatcher is not None and isinstance(f, Dispatcher):
            f = f.py_func
        func_name = f.__name__

        # code scope
        closure_vars = inspect.getclosurevars(f)
        code_scope = dict(closure_vars.nonlocals)
        code_scope.update(dict(closure_vars.globals))

        # separate variables
        analysis = ast_analysis.separate_variables(f)
        variables_for_returns = analysis['variables_for_returns']
        expressions_for_returns = analysis['expressions_for_returns']
        for vi, (key, vars) in enumerate(variables_for_returns.items()):
            variables = []
            for v in vars:
                if len(v) > 1:
                    raise ValueError('Cannot analyze must assignment code line.')
                variables.append(v[0])
            expressions = expressions_for_returns[key]
            var_name = integral.variables[vi]
            DE = sympy_analysis.SingleDiffEq(var_name=var_name,
                                             variables=variables,
                                             expressions=expressions,
                                             derivative_expr=key,
                                             scope=code_scope,
                                             func_name=func_name)
            analyzers.append(DE)

        # others
        all_variables.update(integral.variables)
        all_parameters.update(integral.parameters)
        all_scope.update(code_scope)

    return DynamicModel(analyzers=analyzers,
                        variables=list(all_variables),
                        parameters=list(all_parameters),
                        scopes=all_scope)


class DynamicModel(object):
    def __init__(self, analyzers, variables, parameters, scopes):
        self.analyzers = analyzers
        self.variables = variables
        self.parameters = parameters
        self.scopes = scopes
