# -*- coding: utf-8 -*-

import ast
import inspect
import re
from collections import OrderedDict
from pprint import pprint

from brainpy import backend
from brainpy import errors
from brainpy import tools
from brainpy.backend import utils
from brainpy.backend.numpy.driver import TensorDSDriver
from brainpy.integrators import constants as diffint_cons
from brainpy.simulation import drivers
from brainpy.simulation.brainobjects import delays

try:
  import numba
  from numba.core.dispatcher import Dispatcher
except ModuleNotFoundError:
  raise errors.BackendNotInstalled('numba')

__all__ = [
  'set_numba_profile',
  'get_numba_profile',

  'NumbaDiffIntDriver',
  'NumbaDSDriver',
]

NUMBA_PROFILE = {
  'nopython': True,
  'fastmath': True,
  'nogil': False,
  'parallel': False
}


def set_numba_profile(**kwargs):
  """Set the compilation options of Numba JIT function.

  Parameters
  ----------
  kwargs : Any
      The arguments, including ``cache``, ``fastmath``,
      ``parallel``, ``nopython``.
  """
  global NUMBA_PROFILE

  if 'fastmath' in kwargs:
    NUMBA_PROFILE['fastmath'] = kwargs.pop('fastmath')
  if 'nopython' in kwargs:
    NUMBA_PROFILE['nopython'] = kwargs.pop('nopython')
  if 'nogil' in kwargs:
    NUMBA_PROFILE['nogil'] = kwargs.pop('nogil')
  if 'parallel' in kwargs:
    NUMBA_PROFILE['parallel'] = kwargs.pop('parallel')


def get_numba_profile():
  """Get the compilation setting of numba JIT function.

  Returns
  -------
  numba_setting : dict
      Numba setting.
  """
  return NUMBA_PROFILE


class NumbaDiffIntDriver(drivers.BaseDiffIntDriver):
  def build(self, *args, **kwargs):
    # code
    code = '\n'.join(self.code_lines)
    if self.show_code:
      print(code)
      print()
      pprint(self.code_scope)
      print()

    # jit original functions
    has_jitted = isinstance(self.code_scope['f'], Dispatcher)
    if not has_jitted:
      if self.func_name.startswith(diffint_cons.ODE_PREFIX):
        self.code_scope['f'] = numba.jit(**get_numba_profile())(self.code_scope['f'])
      elif self.func_name.startswith(diffint_cons.SDE_PREFIX):
        self.code_scope['f'] = numba.jit(**get_numba_profile())(self.code_scope['f'])
        self.code_scope['g'] = numba.jit(**get_numba_profile())(self.code_scope['g'])
      else:
        raise NotImplementedError

    # compile
    exec(compile(code, '', 'exec'), self.code_scope)

    # attribute assignment
    new_f = self.code_scope[self.func_name]
    for key, value in self.uploads.items():
      setattr(new_f, key, value)
    if not has_jitted:
      new_f = numba.jit(**get_numba_profile())(new_f)
    return new_f


class _CPUReader(ast.NodeVisitor):
  """The following tasks should be carried out:

  - Find all expressions, including Assign, AugAssign, For loop, If-else condition.
  - Find all delay push and pull.

  """

  def __init__(self, host):
    self.lefts = []
    self.rights = []
    self.lines = []
    self.visited_nodes = set()

    self.host = host
    self.visited_calls = {}  # will focused on delay calls

  def visit_Assign(self, node, level=0):
    if node not in self.visited_nodes:
      prefix = '  ' * level
      expr = tools.ast2code(ast.fix_missing_locations(node.value))
      targets = []
      for target in node.targets:
        targets.append(tools.ast2code(ast.fix_missing_locations(target)))
      _target = ' = '.join(targets)

      self.rights.append(expr)
      self.lefts.append(_target)
      self.lines.append(f'{prefix}{_target} = {expr}')

      self.visited_nodes.add(node)

    self.generic_visit(node)

  def visit_AugAssign(self, node, level=0):
    if node not in self.visited_nodes:
      prefix = '  ' * level
      op = tools.ast2code(ast.fix_missing_locations(node.op))
      expr = tools.ast2code(ast.fix_missing_locations(node.value))
      target = tools.ast2code(ast.fix_missing_locations(node.target))

      self.lefts.append(target)
      self.rights.append(f'{target} {op} {expr}')
      self.lines.append(f"{prefix}{target} = {target} {op} {expr}")

      self.visited_nodes.add(node)

    self.generic_visit(node)

  def visit_AnnAssign(self, node):
    raise NotImplementedError('Do not support an assignment with '
                              'a type annotation in Numba backend.')

  def visit_node_not_assign(self, node, level=0):
    if node not in self.visited_nodes:
      prefix = '  ' * level
      expr = tools.ast2code(ast.fix_missing_locations(node))
      self.lines.append(f'{prefix}{expr}')
      self.lefts.append('')
      self.rights.append(expr)
      self.visited_nodes.add(node)

    self.generic_visit(node)

  def visit_Assert(self, node, level=0):
    self.visit_node_not_assign(node, level)

  def visit_Expr(self, node, level=0):
    self.visit_node_not_assign(node, level)

  def visit_Expression(self, node, level=0):
    self.visit_node_not_assign(node, level)

  def visit_Return(self, node, level=0):
    self.visit_node_not_assign(node, level)

  def visit_content_in_condition_control(self, node, level):
    if isinstance(node, ast.Expr):
      self.visit_Expr(node, level)
    elif isinstance(node, ast.Assert):
      self.visit_Assert(node, level)
    elif isinstance(node, ast.Assign):
      self.visit_Assign(node, level)
    elif isinstance(node, ast.AugAssign):
      self.visit_AugAssign(node, level)
    elif isinstance(node, ast.If):
      self.visit_If(node, level)
    elif isinstance(node, ast.For):
      self.visit_For(node, level)
    elif isinstance(node, ast.While):
      self.visit_While(node, level)
    elif isinstance(node, ast.Call):
      self.visit_Call(node, level)
    elif isinstance(node, ast.Raise):
      self.visit_Raise(node, level)
    elif isinstance(node, ast.Return):
      self.visit_Return(node, level)
    else:
      code = tools.ast2code(ast.fix_missing_locations(node))
      raise errors.CodeError(f'BrainPy does not support {type(node)} '
                             f'in Numba backend.\n\n{code}')

  def visit_attr(self, node):
    if isinstance(node, ast.Attribute):
      r = self.visit_attr(node.value)
      return [node.attr] + r
    elif isinstance(node, ast.Name):
      return [node.id]
    else:
      raise ValueError

  def visit_Call(self, node, level=0):
    if getattr(node, 'starargs', None) is not None:
      raise ValueError("Variable number of arguments not supported")
    if getattr(node, 'kwargs', None) is not None:
      raise ValueError("Keyword arguments not supported")

    if node in self.visited_calls:
      return node

    calls = self.visit_attr(node.func)
    calls = calls[::-1]

    # get the object and the function
    if calls[0] not in backend.CLASS_KEYWORDS:
      return node
    obj = self.host
    for data in calls[1:-1]:
      obj = getattr(obj, data)
    obj_func = getattr(obj, calls[-1])

    # get function arguments
    args = []
    for arg in node.args:
      args.append(tools.ast2code(ast.fix_missing_locations(arg)))
    kw_args = OrderedDict()
    for keyword in node.keywords:
      kw_args[keyword.arg] = tools.ast2code(ast.fix_missing_locations(keyword.value))

    # TASK 1 : extract delay push and delay pull
    # ------
    # Replace the delay function call to the delay_data
    # index. In such a way, delay function will be removed.
    # ------

    if calls[-1] in ['push', 'pull'] and isinstance(obj, delays.ConstantDelay) and callable(obj_func):
      dvar4call = '.'.join(calls[0:-1])
      uniform_delay = getattr(obj, 'uniform_delay')
      if calls[-1] == 'push':
        data_need_pass = [f'{dvar4call}.delay_data', f'{dvar4call}.delay_in_idx']
        idx_or_val = kw_args['idx_or_val'] if len(args) == 0 else args[0]
        if len(args) + len(kw_args) == 1:
          rep_expression = f'{dvar4call}.delay_data[{dvar4call}.delay_in_idx] = {idx_or_val}'
        elif len(args) + len(kw_args) == 2:
          value = kw_args['value'] if len(args) <= 1 else args[1]
          if uniform_delay:
            rep_expression = f'{dvar4call}.delay_data[{dvar4call}.delay_in_idx, {idx_or_val}] = {value}'
          else:
            rep_expression = f'{dvar4call}.delay_data[{dvar4call}.delay_in_idx[{idx_or_val}], {idx_or_val}] = {value}'
        else:
          raise errors.CodeError(f'Cannot analyze the code: \n\n'
                                 f'{tools.ast2code(ast.fix_missing_locations(node))}')
      else:
        data_need_pass = [f'{dvar4call}.delay_data', f'{dvar4call}.delay_out_idx']
        if len(args) + len(kw_args) == 0:
          rep_expression = f'{dvar4call}.delay_data[{dvar4call}.delay_out_idx]'
        elif len(args) + len(kw_args) == 1:
          idx = kw_args['idx'] if len(args) == 0 else args[0]
          if uniform_delay:
            rep_expression = f'{dvar4call}.delay_data[{dvar4call}.delay_out_idx, {idx}]'
          else:
            rep_expression = f'{dvar4call}.delay_data[{dvar4call}.delay_out_idx[{idx}], {idx}]'
        else:
          raise errors.CodeError(f'Cannot analyze the code: \n\n'
                                 f'{tools.ast2code(ast.fix_missing_locations(node))}')

      org_call = tools.ast2code(ast.fix_missing_locations(node))
      self.visited_calls[node] = dict(type=calls[-1],
                                      org_call=org_call,
                                      rep_call=rep_expression,
                                      data_need_pass=data_need_pass)

    self.generic_visit(node)

  def visit_If(self, node, level=0):
    if node not in self.visited_nodes:
      # If condition
      prefix = '  ' * level
      compare = tools.ast2code(ast.fix_missing_locations(node.test))
      self.rights.append(f'if {compare}:')
      self.lines.append(f'{prefix}if {compare}:')

      # body
      for expr in node.body:
        self.visit_content_in_condition_control(expr, level + 1)

      # elif
      while node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
        node = node.orelse[0]
        compare = tools.ast2code(ast.fix_missing_locations(node.test))
        self.lines.append(f'{prefix}elif {compare}:')
        for expr in node.body:
          self.visit_content_in_condition_control(expr, level + 1)

      # else:
      if len(node.orelse) > 0:
        self.lines.append(f'{prefix}else:')
        for expr in node.orelse:
          self.visit_content_in_condition_control(expr, level + 1)

      self.visited_nodes.add(node)

    self.generic_visit(node)

  def visit_For(self, node, level=0):
    if node not in self.visited_nodes:
      prefix = '  ' * level
      # target
      target = tools.ast2code(ast.fix_missing_locations(node.target))
      # iter
      iter = tools.ast2code(ast.fix_missing_locations(node.iter))
      self.rights.append(f'{target} in {iter}')
      self.lines.append(prefix + f'for {target} in {iter}:')
      # body
      for expr in node.body:
        self.visit_content_in_condition_control(expr, level + 1)
      # else
      if len(node.orelse) > 0:
        self.lines.append(prefix + 'else:')
        for expr in node.orelse:
          self.visit_content_in_condition_control(expr, level + 1)

      self.visited_nodes.add(node)
    self.generic_visit(node)

  def visit_While(self, node, level=0):
    if node not in self.visited_nodes:
      prefix = '  ' * level
      # test
      test = tools.ast2code(ast.fix_missing_locations(node.test))
      self.rights.append(test)
      self.lines.append(prefix + f'while {test}:')
      # body
      for expr in node.body:
        self.visit_content_in_condition_control(expr, level + 1)
      # else
      if len(node.orelse) > 0:
        self.lines.append(prefix + 'else:')
        for expr in node.orelse:
          self.visit_content_in_condition_control(expr, level + 1)

      self.visited_nodes.add(node)
    self.generic_visit(node)

  def visit_Raise(self, node, level=0):
    if node not in self.visited_nodes:
      prefix = '  ' * level
      line = tools.ast2code(ast.fix_missing_locations(node))
      self.lines.append(prefix + line)

      self.visited_nodes.add(node)
    self.generic_visit(node)

  def visit_Try(self, node):
    raise errors.CodeError('Do not support "try" handler in Numba backend.')

  def visit_With(self, node):
    raise errors.CodeError('Do not support "with" block in Numba backend.')

  def visit_Delete(self, node):
    raise errors.CodeError('Do not support "del" operation in Numba backend.')


def _analyze_step_func(host, f):
  """Analyze the step functions in a population.

  Parameters
  ----------
  host : DynamicSystem
      The data and the function host.
  f : callable
      The step function.

  Returns
  -------
  results : dict
      The code string of the function, the code scope,
      the data need pass into the arguments,
      the data need return.
  """
  code_string = tools.deindent(inspect.getsource(f)).strip()
  tree = ast.parse(code_string)

  # arguments
  # ---------
  args = tools.ast2code(ast.fix_missing_locations(tree.body[0].args)).split(',')

  # AST analysis
  # ------------
  formatter = _CPUReader(host=host)
  formatter.visit(tree)

  # data assigned by self.xx in line right
  # ---
  self_data_in_right = []
  if args[0] in backend.CLASS_KEYWORDS:
    code = ', \n'.join(formatter.rights)
    self_data_in_right = re.findall('\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b', code)
    self_data_in_right = list(set(self_data_in_right))

  # data assigned by self.xxx in line left
  # ---
  code = ', \n'.join(formatter.lefts)
  self_data_without_index_in_left = []
  self_data_with_index_in_left = []
  if args[0] in backend.CLASS_KEYWORDS:
    class_p1 = '\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b'
    self_data_without_index_in_left = set(re.findall(class_p1, code))
    # class_p2 = '(\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*)\\[.*\\]'
    class_p2 = '(\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*)\\['
    self_data_with_index_in_left = set(re.findall(class_p2, code))  # - self_data_without_index_in_left
    # self_data_with_index_in_left = set(re.findall(class_p2, code)) - self_data_without_index_in_left
    self_data_with_index_in_left = list(self_data_with_index_in_left)
    self_data_without_index_in_left = list(self_data_without_index_in_left)

  # code scope
  # ----------
  closure_vars = inspect.getclosurevars(f)
  code_scope = dict(closure_vars.nonlocals)
  code_scope.update(closure_vars.globals)

  # final
  # -----
  self_data_in_right = sorted(self_data_in_right)
  self_data_without_index_in_left = sorted(self_data_without_index_in_left)
  self_data_with_index_in_left = sorted(self_data_with_index_in_left)

  analyzed_results = {
    'delay_call': formatter.visited_calls,
    'code_string': '\n'.join(formatter.lines),
    'code_scope': code_scope,
    'self_data_in_right': self_data_in_right,
    'self_data_without_index_in_left': self_data_without_index_in_left,
    'self_data_with_index_in_left': self_data_with_index_in_left,
  }

  return analyzed_results


def _class2func(cls_func, host, func_name=None, show_code=False):
  """Transform the function in a class into the ordinary function which is
  compatible with the Numba JIT compilation.

  Parameters
  ----------
  cls_func : function
      The function of the instantiated class.
  func_name : str
      The function name. If not given, it will get the function by `cls_func.__name__`.
  show_code : bool
      Whether show the code.

  Returns
  -------
  new_func : function
      The transformed function.
  """
  class_arg, arguments = utils.get_args(cls_func)
  func_name = cls_func.__name__ if func_name is None else func_name
  host_name = host.name

  # get code analysis
  # --------
  analyzed_results = _analyze_step_func(host=host, f=cls_func)
  delay_call = analyzed_results['delay_call']
  main_code = analyzed_results['code_string']
  code_scope = analyzed_results['code_scope']
  self_data_in_right = analyzed_results['self_data_in_right']
  self_data_without_index_in_left = analyzed_results['self_data_without_index_in_left']
  self_data_with_index_in_left = analyzed_results['self_data_with_index_in_left']
  num_indent = utils.get_num_indent(main_code)
  data_need_pass = sorted(list(set(self_data_in_right + self_data_with_index_in_left)))
  data_need_return = self_data_without_index_in_left

  # arguments 1: the function intrinsic needed arguments
  # -----------
  calls = []
  for arg in arguments:
    if hasattr(host, arg):
      calls.append(f'{host_name}.{arg}')
    elif arg in backend.SYSTEM_KEYWORDS:
      calls.append(arg)
    else:
      raise errors.ModelDefError(f'Step function "{func_name}" of {host} '
                                 f'define an unknown argument "{arg}" which is not '
                                 f'an attribute of {host} nor the system keywords '
                                 f'{backend.SYSTEM_KEYWORDS}.')

  # reprocess delay function
  # -----------
  replaces_early = {}
  if len(delay_call) > 0:
    for delay_ in delay_call.values():
      # # method 1: : delay push / delay pull
      # # ------
      # # delay_ = dict(type=calls[-1],
      # #               args=args,
      # #               keywords=keywords,
      # #               kws_append=kws_append,
      # #               func=func,
      # #               org_call=org_call,
      # #               rep_call=rep_call,
      # #               data_need_pass=data_need_pass)
      # if delay_['type'] == 'push':
      #     if len(delay_['args'] + delay_['keywords']) == 2:
      #         func = numba.njit(delay.push_type2)
      #     elif len(delay_['args'] + delay_['keywords']) == 1:
      #         func = numba.njit(delay.push_type1)
      #     else:
      #         raise ValueError(f'Unknown delay push. {delay_}')
      # else:
      #     if len(delay_['args'] + delay_['keywords']) == 1:
      #         func = numba.njit(delay.pull_type1)
      #     elif len(delay_['args'] + delay_['keywords']) == 0:
      #         func = numba.njit(delay.pull_type0)
      #     else:
      #         raise ValueError(f'Unknown delay pull. {delay_}')
      # delay_call_name = delay_['func']
      # if delay_call_name in data_need_pass:
      #     data_need_pass.remove(delay_call_name)
      # code_scope[utils.attr_replace(delay_call_name)] = func
      # data_need_pass.extend(delay_['data_need_pass'])
      # replaces_early[delay_['org_call']] = delay_['rep_call']
      # replaces_later[delay_call_name] = utils.attr_replace(delay_call_name)

      # method 2: : delay push / delay pull
      # ------
      # delay_ = dict(type=calls[-1],
      #               args=args,
      #               kws_append=kws_append,
      #               func=func,
      #               org_call=org_call,
      #               rep_call=rep_expression,
      #               data_need_pass=data_need_pass)
      data_need_pass.extend(delay_['data_need_pass'])
      replaces_early[delay_['org_call']] = delay_['rep_call']
  for target, dest in replaces_early.items():
    main_code = main_code.replace(target, dest)

  # arguments 2: data need pass
  # -----------
  replaces_later = {}
  new_args = arguments + []
  for data in sorted(set(data_need_pass)):
    splits = data.split('.')
    replaces_later[data] = utils.attr_replace(data)
    obj = host
    for attr in splits[1:]:
      obj = getattr(obj, attr)
    if callable(obj):
      code_scope[utils.attr_replace(data)] = obj
      continue
    new_args.append(utils.attr_replace(data))
    calls.append('.'.join([host_name] + splits[1:]))

  # data need return
  # -----------
  assigns = []
  returns = []
  for data in data_need_return:
    splits = data.split('.')
    assigns.append('.'.join([host_name] + splits[1:]))
    returns.append(utils.attr_replace(data))
    replaces_later[data] = utils.attr_replace(data)

  # code scope
  code_scope[host_name] = host

  # codes
  header = f'def new_{func_name}({", ".join(new_args)}):\n'
  main_code = header + tools.indent(main_code, spaces_per_tab=2)
  if len(returns):
    main_code += f'\n{" " * num_indent + "  "}return {", ".join(returns)}'
  main_code = tools.word_replace(main_code, replaces_later)
  if show_code:
    print(main_code)
    print(code_scope)
    print()

  # recompile
  exec(compile(main_code, '', 'exec'), code_scope)
  func = code_scope[f'new_{func_name}']
  func = numba.jit(**NUMBA_PROFILE)(func)
  return func, calls, assigns


class NumbaDSDriver(TensorDSDriver):
  def get_steps_func(self, show_code=False):
    for func_name, step in self.target.steps.items():
      if hasattr(step, '__self__'):
        host = step.__self__
      else:
        host = self.target
      host_name = getattr(host, 'name')

      # transform the class bounded function to the static normal function
      func, calls, assigns = _class2func(cls_func=step,
                                         host=host,
                                         func_name=func_name,
                                         show_code=show_code)
      setattr(host, f'new_{func_name}', func)

      # assignments
      assignment_line = ''
      if len(assigns):
        assignment_line = f'{", ".join(assigns)} = '
      line = f'{assignment_line}{host.name}.new_{func_name}({", ".join(calls)})'

      # format codes according to interval time
      line_calls, code_scope = self.step_lines_by_interval(
        step=step,
        lines=[line, ],
        interval_name=f'{host_name}_{func_name}_interval',
        code_scope={host.name: host})

      # final
      self.formatted_funcs[func_name] = {'func': func,
                                         'scope': code_scope,
                                         'call': line_calls}
