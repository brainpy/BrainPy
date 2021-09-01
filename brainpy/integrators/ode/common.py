# -*- coding: utf-8 -*-

from brainpy import errors
from brainpy.integrators import constants
from pprint import pprint


_ODE_UNKNOWN_NO = 0


def f_names(f):
  func_name = constants.unique_name('ode')
  if f.__name__.isidentifier():
    func_name += '_' + f.__name__
  return func_name


def step(class_kw, vars, dt_var, A, C, code_lines, other_args):
  # steps
  for si, sval in enumerate(A):
    # k-step arguments
    k_args = []
    for v in vars:
      k_arg = f'{v}'
      for j, sv in enumerate(sval):
        if sv not in [0., '0.0', '0.', '0']:
          if sv in ['1.0', '1.', '1', 1.]:
            k_arg += f' + {dt_var} * d{v}_k{j + 1}'
          else:
            k_arg += f' + {dt_var} * d{v}_k{j + 1} * {sv}'
      if k_arg != v:
        name = f'k{si + 1}_{v}_arg'
        code_lines.append(f'  {name} = {k_arg}')
        k_args.append(name)
      else:
        k_args.append(v)

    t_arg = 't'
    if C[si] not in [0., '0.', '0']:
      if C[si] in ['1.', '1', 1.]:
        t_arg += f' + {dt_var}'
      else:
        t_arg += f' + {dt_var} * {C[si]}'
      name = f'k{si + 1}_t_arg'
      code_lines.append(f'  {name} = {t_arg}')
      k_args.append(name)
    else:
      k_args.append(t_arg)

    # k-step derivative names
    k_derivatives = [f'd{v}_k{si + 1}' for v in vars]

    # k-step code line
    code_lines.append(f'  {", ".join(k_derivatives)} = f('
                      f'{", ".join(class_kw + k_args + other_args[1:])})')


def update(vars, dt_var, B, code_lines):
  return_args = []
  for v in vars:
    result = v
    for i, b1 in enumerate(B):
      if b1 not in [0., '0.', '0']:
        result += f' + d{v}_k{i + 1} * {dt_var} * {b1}'
    code_lines.append(f'  {v}_new = {result}')
    return_args.append(f'{v}_new')
  return return_args


def compile_and_assign_attrs(raw_func,
                             code_lines,
                             code_scope,
                             func_name,
                             variables,
                             parameters,
                             dt,
                             var_type,
                             show_code=False):
  if not isinstance(raw_func, dict):
    raise errors.IntegratorError('"raw_func" must be a dict of functions.')

  code_scope_old = {key: val for key, val in code_scope.items()}

  # compile functions
  code = '\n'.join(code_lines)
  if show_code:
    print(code)
    print()
    pprint(code_scope)
    print()
  exec(compile(code, '', 'exec'), code_scope)
  new_f = code_scope[func_name]

  # assign values
  new_f.brainpy_data = dict(raw_func=raw_func,
                            code_lines=code_lines,
                            code_scope=code_scope_old,
                            variables=variables,
                            parameters=parameters,
                            dt=dt,
                            func_name=func_name,
                            var_type=var_type)

  # new_f.raw_func = raw_func
  # new_f.code_lines = code_lines
  # new_f.code_scope = code_scope_old
  # new_f.variables = variables
  # new_f.parameters = parameters
  # new_f.dt = dt
  # new_f.var_type = var_type

  return new_f
