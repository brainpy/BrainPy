# -*- coding: utf-8 -*-

from .. import tools
from .. import errors


class NeuronDynamicsAnalyzer(object):
    def __init__(self,
                 model,
                 target_vars,
                 fixed_vars,
                 pars_update=None):
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
                raise errors.ModelUseError(f'target "{key}" is not a dynamical variable.')
            integrator = var2eq[key]
            diff_eq = integrator.diff_eq
            sub_exprs = diff_eq.get_f_expressions(substitute_vars=list(self.target_vars.keys()))
            old_exprs = diff_eq.get_f_expressions(substitute_vars=None)
            self.target_eqs[key] = tools.DictPlus(sub_exprs=sub_exprs,
                                                  old_exprs=old_exprs,
                                                  diff_eq=diff_eq,
                                                  func_name=diff_eq.func_name)


