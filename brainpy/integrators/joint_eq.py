# -*- coding: utf-8 -*-

import inspect

from brainpy import errors
from brainpy.base import Collector

__all__ = [
  'JointEq',
]


def _get_args(f):
  """Get the function arguments"""
  args = []
  kwargs = {}
  for name, par in inspect.signature(f).parameters.items():
    if par.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
      if par.default is inspect._empty:
        args.append(par.name)
      else:
        kwargs[par.name] = par.default
    elif par.kind is inspect.Parameter.VAR_POSITIONAL:
      raise errors.DiffEqError(f'{JointEq.__name__} does not support VAR_POSITIONAL parameters '
                               f'*{par.name} (error in {f}).')
    elif par.kind is inspect.Parameter.KEYWORD_ONLY:
      raise errors.DiffEqError(f'{JointEq.__name__} does not support KEYWORD_ONLY parameters, '
                               f'e.g., *  (error in {f}).')
    elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
      raise errors.DiffEqError(f'{JointEq.__name__} does not support POSITIONAL_ONLY parameters, '
                               'e.g., /  (error in {f}).')
    elif par.kind is inspect.Parameter.VAR_KEYWORD:
      raise errors.DiffEqError(f'{JointEq.__name__} does not support VAR_KEYWORD '
                               f'arguments **{par.name} (error in {f}).')
    else:
      raise errors.DiffEqError(f'Unknown argument type: {par.kind}')

  # variables
  vars = []
  for a in args:
    if a == 't':
      break
    vars.append(a)
  else:
    raise ValueError('Do not find time variable "t".')

  return vars, args, kwargs


def _std_func(f, all_vars: list):
  f_vars, f_args, f_kwargs = _get_args(f)

  def call(t, *vars, **args_and_kwargs):
    params = dict(t=t)
    for var in f_vars:
      params[var] = vars[all_vars.index(var)]
    for par in f_args[len(f_vars) + 1:]:
      if par in args_and_kwargs:
        params[par] = args_and_kwargs[par]
      else:
        if par not in all_vars:
          raise errors.DiffEqError(f'Missing {par} during the functional call of {f}.')
        params[par] = vars[all_vars.index(par)]
    for par, value in f_kwargs.items():
      if par in args_and_kwargs:
        params[par] = args_and_kwargs[par]
    return f(**params)

  return call


class JointEq(object):
  """Make a joint equation from multiple derivation functions.

  For example, we have an Izhikevich neuron model,

  >>> a, b = 0.02, 0.20
  >>> dV = lambda V, t, u, Iext: 0.04 * V * V + 5 * V + 140 - u + Iext
  >>> du = lambda u, t, V: a * (b * V - u)

  If we make numerical solver for each derivative function, they will be solved independently.

  >>> import brainpy as bp
  >>> bp.odeint(dV, method='rk2', show_code=True)
  def brainpy_itg_of_ode0(V, t, u, Iext, dt=0.1):
    dV_k1 = f(V, t, u, Iext)
    k2_V_arg = V + dt * dV_k1 * 0.6666666666666666
    k2_t_arg = t + dt * 0.6666666666666666
    dV_k2 = f(k2_V_arg, k2_t_arg, u, Iext)
    V_new = V + dV_k1 * dt * 0.25 + dV_k2 * dt * 0.75
    return V_new

  As you see in the output code, "dV_k2" is evaluated by :math:`f(V_{k2}, u)`.
  If you want to solve the above coupled equation jointly, i.e., evalute "dV_k2"
  with :math:`f(V_{k2}, u_{k2})`, you can use :py:class:`brainpy.JointEq`
  to emerge the above two derivative equations into a joint equation, so that
  they will be numerically solved together. Let's see the difference:

  >>> eq = bp.JointEq(eqs=(dV, du))
  >>> bp.odeint(eq, method='rk2', show_code=True)
  def brainpy_itg_of_ode0_joint_eq(V, u, t, Iext, dt=0.1):
    dV_k1, du_k1 = f(V, u, t, Iext)
    k2_V_arg = V + dt * dV_k1 * 0.6666666666666666
    k2_u_arg = u + dt * du_k1 * 0.6666666666666666
    k2_t_arg = t + dt * 0.6666666666666666
    dV_k2, du_k2 = f(k2_V_arg, k2_u_arg, k2_t_arg, Iext)
    V_new = V + dV_k1 * dt * 0.25 + dV_k2 * dt * 0.75
    u_new = u + du_k1 * dt * 0.25 + du_k2 * dt * 0.75
    return V_new, u_new

  :py:class:`brainpy.JointEq` supports make nested ``JointEq``, which means
  the instance of ``JointEq`` can be an element to compose a new ``JointEq``.

  >>> dw = lambda w, t, V: a * (b * V - w)
  >>> eq2 = bp.JointEq(eqs=(eq, dw))


  Parameters
  ----------
  *eqs :
    The elements of derivative function to compose.
  """

  def _check_eqs(self, eqs):
    for eq in eqs:
      if isinstance(eq, (list, tuple)):
        for a in self._check_eqs(eq):
          yield a
      elif callable(eq):
        yield eq
      else:
        raise errors.DiffEqError(f'Elements in "eqs" only supports callable function, but got {eq}.')

  def __init__(self, *eqs):
    eqs = list(self._check_eqs(eqs))

    # variables in equations
    self.vars_in_eqs = []
    vars_in_eqs = []
    for eq in eqs:
      vars, _, _ = _get_args(eq)
      for var in vars:
        if var in vars_in_eqs:
          raise errors.DiffEqError(f'Variable "{var}" has been used, however we got a same '
                                   f'variable name in {eq}. Please change another name.')
      vars_in_eqs.extend(vars)
      self.vars_in_eqs.append(vars)

    # arguments in equations
    self.args_in_eqs = []
    all_arg_pars = []
    all_kwarg_pars = dict()
    for eq in eqs:
      vars, args, kwargs = _get_args(eq)
      self.args_in_eqs.append(args + list(kwargs.keys()))
      for par in args[len(vars) + 1:]:
        if (par not in vars_in_eqs) and (par not in all_arg_pars) and (par not in all_kwarg_pars):
          all_arg_pars.append(par)
      for key, value in kwargs.items():
        if key in all_kwarg_pars and value != all_kwarg_pars[key]:
          raise errors.DiffEqError(f'We got two different default value of "{key}": '
                                   f'{all_kwarg_pars[key]} != {value}')
        elif (key not in vars_in_eqs) and (key not in all_arg_pars):
          all_kwarg_pars[key] = value
        else:
          raise errors.DiffEqError

    # # variable names provided
    # if not isinstance(variables, (tuple, list)):
    #   raise errors.DiffEqError(f'"variables" must be a list/tuple of str, but we got {variables}')
    # for v in variables:
    #   if not isinstance(v, str):
    #     raise errors.DiffEqError(f'"variables" must be a list/tuple of str, but we got {v} in "variables"')
    # if len(vars_in_eqs) != len(variables):
    #   raise errors.DiffEqError(f'We detect {len(vars_in_eqs)} variables "{vars_in_eqs}" '
    #                            f'in the provided equations. However, the used provided '
    #                            f'"variables" have {len(variables)} variables '
    #                            f'"{variables}".')
    # if len(set(vars_in_eqs) - set(variables)) != 0:
    #   raise errors.DiffEqError(f'We detect there are variable "{vars_in_eqs}" in the provided '
    #                            f'equations, while the user provided variables "{variables}" '
    #                            f'is not the same.')

    # finally
    self.eqs = eqs
    # self.variables = variables
    self.arg_keys = vars_in_eqs + ['t'] + all_arg_pars
    self.kwarg_keys = list(all_kwarg_pars.keys())
    self.kwargs = all_kwarg_pars
    parameters = [inspect.Parameter(vp, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                  for vp in self.arg_keys]
    parameters.extend([inspect.Parameter(k,
                                         kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                         default=all_kwarg_pars[k])
                       for k in self.kwarg_keys])
    signature = inspect.signature(eqs[0])
    self.__signature__ = signature.replace(parameters=parameters)
    self.__name__ = 'joint_eq'

  def __call__(self, *args, **kwargs):
    # format arguments
    params_in = Collector()
    for i, arg in enumerate(args):
      if i < len(self.arg_keys):
        params_in[self.arg_keys[i]] = arg
      else:
        params_in[self.kwarg_keys[i - len(self.arg_keys)]] = arg
    params_in.update(kwargs)

    # call equations
    results = []
    for i, eq in enumerate(self.eqs):
      r = eq(**{arg: params_in[arg] for arg in self.args_in_eqs[i]})
      if isinstance(r, (list, tuple)):
        results.extend(list(r))
      else:
        results.append(r)
    return results
