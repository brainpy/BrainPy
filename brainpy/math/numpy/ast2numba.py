# -*- coding: utf-8 -*-


"""
TODO: enable code debug and error report; See https://github.com/numba/numba/issues/7370
"""

import ast
import inspect
import re
from pprint import pprint

import numba
import numba.misc.help.inspector as inspector
import numpy as np
from numba.core.dispatcher import Dispatcher

from brainpy import errors, math, tools
from brainpy.base.base import Base
from brainpy.base.collector import Collector
from brainpy.base.function import Function
from brainpy.math import profile

DE_INT = DynamicSystem = Container = None

__all__ = [
  'jit',
]


def jit(obj_or_fun, show_code=False, **jit_setting):
  global DE_INT
  if DE_INT is None:
    from brainpy.integrators.constants import DE_INT

  if callable(obj_or_fun):
    # Function
    if isinstance(obj_or_fun, Function):
      return jit_Func(obj_or_fun, show_code=show_code, **jit_setting)

    # Base
    elif isinstance(obj_or_fun, Base):
      return jit_Base(func=obj_or_fun.__call__, host=obj_or_fun,
                      show_code=show_code, **jit_setting)

      # integrator
    elif hasattr(obj_or_fun, '__name__') and obj_or_fun.__name__.startswith(DE_INT):
      return jit_integrator(intg=obj_or_fun, show_code=show_code, **jit_setting)

    # bounded method
    elif hasattr(obj_or_fun, '__self__') and isinstance(obj_or_fun.__self__, Base):
      return jit_Base(func=obj_or_fun, host=obj_or_fun.__self__,
                      show_code=show_code, **jit_setting)

    else:
      # native function
      if not isinstance(obj_or_fun, Dispatcher):
        return numba.jit(obj_or_fun, **jit_setting)
      else:
        # numba function
        return obj_or_fun

  else:
    return jit_DS(obj_or_fun, show_code=show_code, **jit_setting)


def jit_DS(obj_or_fun, show_code=False, **jit_setting):
  global DynamicSystem
  if DynamicSystem is None:
    from brainpy.simulation.brainobjects.base import DynamicSystem

  if not isinstance(obj_or_fun, DynamicSystem):
    raise errors.UnsupportedError(f'JIT compilation in numpy backend only '
                                  f'supports {Base.__name__}, but we got '
                                  f'{type(obj_or_fun)}.')

  # function analysis
  for key, step in list(obj_or_fun.steps.items()):
    key = key.replace(".", "_")
    r = _jit_func(obj_or_fun=step, show_code=show_code, **jit_setting)
    if r['func'] != step:
      func = _form_final_call(f_org=step, f_rep=r['func'], arg2call=r['arg2call'],
                              arguments=r['arguments'], nodes=r['nodes'],
                              show_code=show_code, name=step.__name__)
      obj_or_fun.steps.replace(key, func)

  # dynamic system
  return obj_or_fun


def jit_integrator(intg, show_code=False, **jit_setting):
  r = _jit_intg_func(intg, show_code=show_code, **jit_setting)
  if len(r['arguments']):
    intg = _form_final_call(f_org=intg, f_rep=r['func'], arg2call=r['arg2call'],
                            arguments=r['arguments'], nodes=r['nodes'],
                            show_code=show_code, name=intg.__name__)
  else:
    intg = r['func']
  return intg


def jit_Func(func, show_code=False, **jit_setting):
  assert isinstance(func, Function)

  r = _jit_Function(func=func, show_code=show_code, **jit_setting)
  if len(r['arguments']):
    func = _form_final_call(f_org=func._f, f_rep=r['func'], arg2call=r['arg2call'],
                            arguments=r['arguments'], nodes=r['nodes'],
                            show_code=show_code, name=func.name + '_call')
  else:
    func = r['func']

  return func


def jit_Base(func, host, show_code=False, **jit_setting):
  r = _jit_cls_func(func, host=host, show_code=show_code, **jit_setting)
  if len(r['arguments']):
    func = _form_final_call(f_org=func, f_rep=r['func'], arg2call=r['arg2call'],
                            arguments=r['arguments'], nodes=r['nodes'],
                            show_code=show_code, name=host.name + '_call')
  else:
    func = r['func']
  return func


def _jit_func(obj_or_fun, show_code=False, **jit_setting):
  global DE_INT
  if DE_INT is None:
    from brainpy.integrators.constants import DE_INT

  if callable(obj_or_fun):
    # integrator
    if hasattr(obj_or_fun, '__name__') and obj_or_fun.__name__.startswith(DE_INT):
      return _jit_intg_func(obj_or_fun, show_code=show_code, **jit_setting)

    # bounded method
    elif hasattr(obj_or_fun, '__self__') and isinstance(obj_or_fun.__self__, Base):
      return _jit_cls_func(obj_or_fun, host=obj_or_fun.__self__,
                           show_code=show_code, **jit_setting)

    # wrapped function
    elif isinstance(obj_or_fun, Function):
      return _jit_Function(obj_or_fun, show_code=show_code, **jit_setting)

    # base class function
    elif isinstance(obj_or_fun, Base):
      return _jit_cls_func(obj_or_fun.__call__, host=obj_or_fun,
                           show_code=show_code, **jit_setting)

    else:
      # native function
      if not isinstance(obj_or_fun, Dispatcher):
        if inspector.inspect_function(obj_or_fun)['numba_type'] is None:
          f = numba.jit(obj_or_fun, **jit_setting)
          return dict(func=f, arguments=set(), arg2call=Collector(), nodes=Collector())
      # numba function or innate supported function
      return dict(func=obj_or_fun, arguments=set(), arg2call=Collector(), nodes=Collector())

  else:
    raise ValueError


def _jit_Function(func, show_code=False, **jit_setting):
  assert isinstance(func, Function)

  # code_scope
  closure_vars = inspect.getclosurevars(func._f)
  code_scope = dict(closure_vars.nonlocals)
  code_scope.update(closure_vars.globals)
  # code
  code = tools.deindent(inspect.getsource(func._f)).strip()
  # arguments
  arguments = set()
  # nodes
  nodes = {v.name: v for v in func._nodes.values()}
  # arg2call
  arg2call = dict()

  for key, node in func._nodes.items():
    code, _arguments, _arg2call, _nodes, code_scope = _analyze_cls_func(
      host=node, code=code, show_code=show_code, code_scope=code_scope,
      self_name=key, pop_self=True, **jit_setting)
    arguments.update(_arguments)
    arg2call.update(_arg2call)
    nodes.update(_nodes)

  # compile new function
  # code, _scope = _add_try_except(code)
  # code_scope.update(_scope)
  if show_code:
    print(code)
    print()
    pprint(code_scope)
    print()
  exec(compile(code, '', 'exec'), code_scope)
  func = code_scope[func._f.__name__]
  func = numba.jit(func, **jit_setting)

  # returns
  return dict(func=func, arguments=arguments, arg2call=arg2call, nodes=nodes)


def _jit_cls_func(f, code=None, host=None, show_code=False, **jit_setting):
  global Container
  if Container is None:
    from brainpy.simulation.brainobjects.base import Container

  host = (host or f.__self__)

  # data to return
  arguments = set()
  arg2call = dict()
  nodes = Collector()
  nodes[host.name] = host

  # step function of Container
  if isinstance(host, Container):
    # if f.__name__ != 'update':
    #   raise errors.UnsupportedError(f'Currently, BrainPy only supports compile "update" step '
    #                                 f'function, while we got {f.__name__}: {f}')
    code_lines = []
    code_scope = {}
    for key, step in host.child_steps.items():
      r = _jit_func(obj_or_fun=step, show_code=show_code, **jit_setting)
      # if r['func'] != step:
      arguments.update(r['arguments'])
      arg2call.update(r['arg2call'])
      nodes.update(r['nodes'])
      code_scope[key.replace('.', '_')] = r['func']
      call_args = [f'{arg}={arg}' for arg in sorted(r['arguments'])]
      code_lines.append("{call}(_t, _dt, {args})".format(call=key.replace('.', '_'),
                                                         args=", ".join(call_args)))
      # args=_items2lines(call_args, line_break='\n\t\t\t')))
    code_lines = ['  ' + line for line in code_lines]
    # code_lines.insert(0, f'def {host.name}_update(_t, _dt, {_items2lines(sorted(arguments))}):')
    code_lines.insert(0, f'def {host.name}_update(_t, _dt, {", ".join(sorted(arguments))}):')
    code = '\n'.join(code_lines)
    # code_scope.update(nodes)
    func_name = f'{host.name}_{f.__name__}'

  # step function of normal DynamicSystem
  else:
    code = (code or tools.deindent(inspect.getsource(f)).strip())
    # function name
    func_name = f.__name__
    # code scope
    closure_vars = inspect.getclosurevars(f)
    code_scope = dict(closure_vars.nonlocals)
    code_scope.update(closure_vars.globals)
    code, _arguments, _arg2call, _nodes, code_scope = _analyze_cls_func(
      host=host, code=code, show_code=show_code, code_scope=code_scope, **jit_setting)
    arguments.update(_arguments)
    arg2call.update(_arg2call)
    nodes.update(_nodes)

  # compile new function
  # code, _scope = _add_try_except(code)
  # code_scope.update(_scope)
  if show_code:
    print(code)
    print()
    pprint(code_scope)
    print()
  exec(compile(code, '', 'exec'), code_scope)
  func = code_scope[func_name]
  func = numba.jit(func, **jit_setting)

  # returns
  return dict(func=func, arguments=arguments, arg2call=arg2call, nodes=nodes)


def _jit_intg_func(f, show_code=False, **jit_setting):
  global DynamicSystem
  if DynamicSystem is None:
    from brainpy.simulation.brainobjects.base import DynamicSystem

  # exponential euler methods
  if f.brainpy_data['method'].startswith('exponential'):
    return _jit_cls_func(f=f, code="\n".join(f.brainpy_data['code_lines']),
                         show_code=show_code, **jit_setting)

  # information in the integrator
  func_name = f.brainpy_data['func_name']
  raw_func = f.brainpy_data['raw_func']
  tree = ast.parse('\n'.join(f.brainpy_data['code_lines']))
  code_scope = {key: val for key, val in f.brainpy_data['code_scope'].items()}

  # essential information
  arguments = set()
  arg2call = dict()
  nodes = Collector()

  # jit raw functions
  f_node = None
  remove_self = None
  if hasattr(f, '__self__') and isinstance(f.__self__, DynamicSystem):
    f_node = f.__self__
    _arg = tree.body[0].args.args.pop(0)  # remove "self" arg
    # remove "self" in functional call
    remove_self = _arg.arg

  need_recompile = False
  for key, func in raw_func.items():
    # get node
    func_node = None
    if f_node:
      func_node = f_node
    elif hasattr(func, '__self__') and isinstance(func.__self__, DynamicSystem):
      func_node = func.__self__

    # get new compiled function
    if isinstance(func, Dispatcher):
      continue
    elif func_node:
      need_recompile = True
      r = _jit_cls_func(f=func, host=func_node, show_code=show_code, **jit_setting)
      if len(r['arguments']) or remove_self:
        tree = _replace_func(tree, func_call=key,
                             arg_to_append=r['arguments'],
                             remove_self=remove_self)
      code_scope[key] = r['func']
      arguments.update(r['arguments'])  # update arguments
      arg2call.update(r['arg2call'])  # update arg2call
      nodes.update(r['nodes'])  # update nodes
      nodes[func_node.name] = func_node  # update nodes
    else:
      need_recompile = True
      code_scope[key] = numba.jit(func, **jit_setting)

  if need_recompile:
    tree.body[0].decorator_list.clear()
    tree.body[0].args.args.extend([ast.Name(id=a) for a in sorted(arguments)])
    tree.body[0].args.defaults.extend([ast.Constant(None) for _ in sorted(arguments)])
    code = tools.ast2code(tree)
    # code, _scope = _add_try_except(code)
    # code_scope.update(_scope)
    code_scope_backup = {k: v for k, v in code_scope.items()}
    # compile functions
    if show_code:
      print(code)
      print()
      pprint(code_scope)
      print()
    exec(compile(code, '', 'exec'), code_scope)
    new_f = code_scope[func_name]
    new_f.brainpy_data = {key: val for key, val in f.brainpy_data.items()}
    new_f.brainpy_data['code_lines'] = code.strip().split('\n')
    new_f.brainpy_data['code_scope'] = code_scope_backup
    jit_f = numba.jit(new_f, **jit_setting)
    return dict(func=jit_f, arguments=arguments, arg2call=arg2call, nodes=nodes)
  else:
    return dict(func=f, arguments=arguments, arg2call=arg2call, nodes=nodes)


class FuncTransformer(ast.NodeTransformer):
  """Transform a functional call.

  This class automatically transform a functional call.
  For example, in your original code:

  ... code-block:: python

      def update(self, _t, _dt):
        V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input)
        self.spike[:] = (self.V < self.V_th) * (V >= self.V_th)

  you want to add new arguments ``gNa``, ``gK`` and ``gL`` into the
  function ``self.integral``. Then this Transformer will help you
  automatically do this:

  ... code-block:: python

      def update(self, _t, _dt):
        V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input,
                                   gNa=gNa, gK=gK, gL=gL)
        self.spike[:] = (self.V < self.V_th) * (V >= self.V_th)

  """

  def __init__(self, func_name, arg_to_append, remove_self=None):
    self.func_name = func_name
    self.arg_to_append = sorted(arg_to_append)
    self.remove_self = remove_self

  def visit_Call(self, node, level=0):
    if getattr(node, 'starargs', None) is not None:
      raise ValueError("Variable number of arguments (*args) are not supported")
    if getattr(node, 'kwargs', None) is not None:
      raise ValueError("Keyword arguments (**kwargs) are not supported")

    # get function name
    call = tools.ast2code(node.func)
    if call == self.func_name:
      # args
      args = [self.generic_visit(arg) for arg in node.args]
      # remove self arg
      if self.remove_self:
        if args[0].id == self.remove_self:
          args.pop(0)
      # kwargs
      kwargs = [self.generic_visit(keyword) for keyword in node.keywords]
      # new kwargs
      code = f'f({", ".join([f"{k}={k}" for k in self.arg_to_append])})'
      tree = ast.parse(code)
      new_keywords = tree.body[0].value.keywords
      kwargs.extend(new_keywords)
      # final function
      return ast.Call(func=node.func, args=args, keywords=kwargs)
    return node


def _replace_func(code_or_tree, func_call, arg_to_append, remove_self=None):
  assert isinstance(func_call, str)
  assert isinstance(arg_to_append, (list, tuple, set))

  if isinstance(code_or_tree, str):
    tree = ast.parse(code_or_tree)
  elif isinstance(code_or_tree, ast.Module):
    tree = code_or_tree
  else:
    raise ValueError

  transformer = FuncTransformer(func_name=func_call,
                                arg_to_append=arg_to_append,
                                remove_self=remove_self)
  new_tree = transformer.visit(tree)
  return new_tree


def _analyze_cls_func(host, code, show_code, code_scope, self_name=None, pop_self=True, **jit_setting):
  """

  Parameters
  ----------
  host : Base
    The data host.
  code : str
    The function source code.
  self_name : optional, str
    The class name, like "self", "cls".
  show_code : bool


  Returns
  -------

  """
  arguments, arg2call, nodes = set(), dict(), Collector()

  # arguments
  tree = ast.parse(code)
  if self_name is None:
    self_name = tree.body[0].args.args[0].arg
    # data assigned by self.xx in line right
    if self_name not in profile.CLASS_KEYWORDS:
      raise errors.CodeError(f'BrainPy only support class keyword '
                             f'{profile.CLASS_KEYWORDS}, but we got {self_name}.')
  if pop_self:
    tree.body[0].args.args.pop(0)  # remove "self" etc. class argument
  self_data = re.findall('\\b' + self_name + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b', code)
  self_data = list(set(self_data))

  # analyze variables and functions accessed by the self.xx
  data_to_replace = {}
  for key in self_data:
    split_keys = key.split('.')
    if len(split_keys) < 2:
      raise errors.BrainPyError

    # get target and data
    target = host
    for i in range(1, len(split_keys)):
      next_target = getattr(target, split_keys[i])
      if not isinstance(next_target, Base):
        break
      target = next_target
    else:
      raise errors.BrainPyError
    data = getattr(target, split_keys[i])

    key = '.'.join(split_keys[:i + 1])

    # analyze data
    if isinstance(data, math.Variable):
      arguments.add(f'{target.name}_{split_keys[i]}')
      arg2call[f'{target.name}_{split_keys[i]}'] = f'{target.name}.{split_keys[-1]}.value'
      nodes[target.name] = target
      data_to_replace[key] = f'{target.name}_{split_keys[i]}'  # replace the data
    elif isinstance(data, np.random.RandomState):
      data_to_replace[key] = f'{target.name}_{split_keys[i]}'  # replace the data
      code_scope[f'{target.name}_{split_keys[i]}'] = np.random  # replace RandomState
    elif callable(data):
      assert len(split_keys) == i + 1
      r = _jit_func(obj_or_fun=data, show_code=show_code, **jit_setting)
      if len(r):
        tree = _replace_func(tree, func_call=key, arg_to_append=r['arguments'])
        arguments.update(r['arguments'])
        arg2call.update(r['arg2call'])
        nodes.update(r['nodes'])
        code_scope[f'{target.name}_{split_keys[i]}'] = r['func']
        data_to_replace[key] = f'{target.name}_{split_keys[i]}'  # replace the data
    else:
      code_scope[f'{target.name}_{split_keys[i]}'] = data
      data_to_replace[key] = f'{target.name}_{split_keys[i]}'  # replace the data

  # final code
  tree.body[0].decorator_list.clear()
  tree.body[0].args.args.extend([ast.Name(id=a) for a in sorted(arguments)])
  tree.body[0].args.defaults.extend([ast.Constant(None) for _ in sorted(arguments)])
  code = tools.ast2code(tree)
  code = tools.word_replace(code, data_to_replace, exclude_dot=False)

  return code, arguments, arg2call, nodes, code_scope


def _add_try_except(code):
  splits = re.compile(r'\)\s*?:').split(code)
  if len(splits) == 1:
    raise ValueError(f"Cannot analyze code:\n{code}")

  def_line = splits[0] + '):'
  code_lines = '):'.join(splits[1:])
  code_lines = [line for line in code_lines.split('\n') if line.strip()]
  main_code = tools.deindent("\n".join(code_lines))

  code = def_line + '\n'
  code += '  try:\n'
  code += tools.indent(main_code, num_tabs=2, spaces_per_tab=2)
  code += '\n'
  code += '  except NumbaError:\n'
  code += '    print(_code_)'
  return code, {'NumbaError': numba.errors.NumbaError,
                '_code_': code}


def _items2lines(items, num_each_line=5, separator=', ', line_break='\n\t\t'):
  res = ''
  for item in items[:num_each_line]:
    res += item + separator
  for i in range(num_each_line, len(items), num_each_line):
    res += line_break
    for item in items[i: i + num_each_line]:
      res += item + separator
  return res


def _form_final_call(f_org, f_rep, arg2call, arguments, nodes, show_code=False, name=None):
  cls_kw, reduce_args, org_args = _get_args(f_org)

  name = (name or f_org.__name__)
  code_scope = {key: node for key, node in nodes.items()}
  code_scope[name] = f_rep
  called_args = _items2lines(reduce_args + [f"{a}={arg2call[a]}" for a in sorted(arguments)]).strip()
  code_lines = [f'def new_{name}({", ".join(org_args)}):',
                f'  {name}({called_args.strip()})']

  # compile new function
  code = '\n'.join(code_lines)
  # code, _scope = _add_try_except(code)
  # code_scope.update(_scope)
  if show_code:
    print(code)
    print()
    pprint(code_scope)
    print()
  exec(compile(code, '', 'exec'), code_scope)
  func = code_scope[f'new_{name}']
  return func


def _get_args(f):
  # 1. get the function arguments
  original_args = []
  reduced_args = []

  for name, par in inspect.signature(f).parameters.items():
    if par.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
      reduced_args.append(par.name)
    elif par.kind is inspect.Parameter.VAR_POSITIONAL:
      reduced_args.append(par.name)
    elif par.kind is inspect.Parameter.KEYWORD_ONLY:
      reduced_args.append(par.name)
    elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
      raise errors.DiffEqError('Don not support positional only parameters, e.g., /')
    elif par.kind is inspect.Parameter.VAR_KEYWORD:
      raise errors.DiffEqError(f'Don not support dict of keyword arguments: {str(par)}')
    else:
      raise errors.DiffEqError(f'Unknown argument type: {par.kind}')

    original_args.append(str(par))

  # 2. analyze the function arguments
  #   2.1 class keywords
  class_kw = []
  if original_args[0] in profile.CLASS_KEYWORDS:
    class_kw.append(original_args[0])
    original_args = original_args[1:]
    reduced_args = reduced_args[1:]
  for a in original_args:
    if a.split('=')[0].strip() in profile.CLASS_KEYWORDS:
      raise errors.DiffEqError(f'Class keywords "{a}" must be defined '
                               f'as the first argument.')
  return class_kw, reduced_args, original_args
