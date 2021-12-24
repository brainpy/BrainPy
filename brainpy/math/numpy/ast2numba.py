# -*- coding: utf-8 -*-


"""
TODO: enable code debug and error report; See https://github.com/numba/numba/issues/7370
"""

import ast
import inspect
import logging
import re
from copy import deepcopy
from pprint import pprint

import numpy as np

try:
  import numba
  import numba.misc.help.inspector as inspector
  from numba.core.dispatcher import Dispatcher
except (ImportError, ModuleNotFoundError):
  numba = Dispatcher = inspector = None

from brainpy import errors, math, tools
from brainpy.base.base import Base
from brainpy.base.collector import Collector
from brainpy.base.function import Function
from brainpy.integrators.base import Integrator
from brainpy.simulation.brainobjects.base import DynamicalSystem

logger = logging.getLogger('brainpy.math.numpy.ast2numba')

__all__ = [
  'jit',
]

CLASS_KEYWORDS = ['self', 'cls']


def jit(obj_or_fun, nopython=True, fastmath=True, parallel=False, nogil=False,
        forceobj=False, looplift=True, error_model='python', inline='never',
        boundscheck=None, show_code=False, debug=False):
  """Just-In-Time (JIT) Compilation in NumPy backend.

  JIT compilation in NumPy backend relies on `Numba <http://numba.pydata.org/>`_. However,
  in BrainPy, `brainpy.math.numpy.jit()` can apply to class objects, especially the instance
  of :py:class:`brainpy.DynamicalSystem`.

  If you are using JAX backend, please refer to the JIT compilation in
  JAX backend ``brainpy.math.jit()``.

  Parameters
  ----------
  debug : bool
  obj_or_fun : callable, Base
    The function or the base model to jit compile.
  nopython : bool
    Set to True to disable the use of PyObjects and Python API
    calls. Default value is True.
  fastmath : bool
    In certain classes of applications strict IEEE 754 compliance
    is less important. As a result it is possible to relax some
    numerical rigour with view of gaining additional performance.
    The way to achieve this behaviour in Numba is through the use
    of the ``fastmath`` keyword argument.
  parallel : bool
    Enables automatic parallelization (and related optimizations) for
    those operations in the function known to have parallel semantics.
  nogil : bool
    Whenever Numba optimizes Python code to native code that only
    works on native types and variables (rather than Python objects),
    it is not necessary anymore to hold Python’s global interpreter
    lock (GIL). Numba will release the GIL when entering such a
    compiled function if you passed ``nogil=True``.
  forceobj: bool
    Set to True to force the use of PyObjects for every value.
    Default value is False.
  looplift: bool
    Set to True to enable jitting loops in nopython mode while
    leaving surrounding code in object mode. This allows functions
    to allocate NumPy arrays and use Python objects, while the
    tight loops in the function can still be compiled in nopython
    mode. Any arrays that the tight loop uses should be created
    before the loop is entered. Default value is True.
  error_model: str
    The error-model affects divide-by-zero behavior.
    Valid values are 'python' and 'numpy'. The 'python' model
    raises exception.  The 'numpy' model sets the result to
    *+/-inf* or *nan*. Default value is 'python'.
  inline: str or callable
    The inline option will determine whether a function is inlined
    at into its caller if called. String options are 'never'
    (default) which will never inline, and 'always', which will
    always inline. If a callable is provided it will be called with
    the call expression node that is requesting inlining, the
    caller's IR and callee's IR as arguments, it is expected to
    return Truthy as to whether to inline.
    NOTE: This inlining is performed at the Numba IR level and is in
    no way related to LLVM inlining.
  boundscheck: bool or None
    Set to True to enable bounds checking for array indices. Out
    of bounds accesses will raise IndexError. The default is to
    not do bounds checking. If False, bounds checking is disabled,
    out of bounds accesses can produce garbage results or segfaults.
    However, enabling bounds checking will slow down typical
    functions, so it is recommended to only use this flag for
    debugging. You can also set the NUMBA_BOUNDSCHECK environment
    variable to 0 or 1 to globally override this flag. The default
    value is None, which under normal execution equates to False,
    but if debug is set to True then bounds checking will be
    enabled.
  show_code : bool
    Debugging.
  """

  # checking
  if numba is None:
    raise errors.PackageMissingError('JIT compilation in numpy backend need Numba. '
                                     'Please install numba via: \n'
                                     '>>> pip install numba\n'
                                     '>>> # or \n'
                                     '>>> conda install numba')
  return _jit(obj_or_fun, show_code=show_code,
              nopython=nopython, fastmath=fastmath, parallel=parallel, nogil=nogil,
              forceobj=forceobj, looplift=looplift, error_model=error_model,
              inline=inline, boundscheck=boundscheck, debug=debug)


def _jit(obj_or_fun, show_code=False, **jit_setting):
  if isinstance(obj_or_fun, DynamicalSystem):
    return _jit_DS(obj_or_fun, show_code=show_code, **jit_setting)

  else:
    assert callable(obj_or_fun)

    # integrator
    if isinstance(obj_or_fun, Integrator):
      return _jit_Integrator(intg=obj_or_fun, show_code=show_code, **jit_setting)

    # Function
    if isinstance(obj_or_fun, Function):
      return _jit_Func(obj_or_fun, show_code=show_code, **jit_setting)

    # Base
    elif isinstance(obj_or_fun, Base):
      return _jit_Base(func=obj_or_fun.__call__,
                       host=obj_or_fun,
                       name=obj_or_fun.name + '_call',
                       show_code=show_code, **jit_setting)

    # bounded method
    elif hasattr(obj_or_fun, '__self__') and isinstance(obj_or_fun.__self__, Base):
      return _jit_Base(func=obj_or_fun,
                       host=obj_or_fun.__self__,
                       show_code=show_code,
                       **jit_setting)

    else:
      # native function
      if not isinstance(obj_or_fun, Dispatcher):
        return numba.jit(obj_or_fun, **jit_setting)
      else:
        # numba function
        return obj_or_fun


def _jit_DS(obj_or_fun, show_code=False, **jit_setting):
  if not isinstance(obj_or_fun, DynamicalSystem):
    raise errors.UnsupportedError(f'JIT compilation in numpy backend only '
                                  f'supports {Base.__name__}, but we got '
                                  f'{type(obj_or_fun)}.')
  if not hasattr(obj_or_fun, 'steps'):
    raise errors.BrainPyError(f'Please init this DynamicalSystem {obj_or_fun} first, '
                              f'then apply JIT.')

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


def _jit_Integrator(intg, show_code=False, **jit_setting):
  r = _jit_intg(intg, show_code=show_code, **jit_setting)
  if len(r['arguments']):
    intg = _form_final_call(f_org=intg, f_rep=r['func'], arg2call=r['arg2call'],
                            arguments=r['arguments'], nodes=r['nodes'],
                            show_code=show_code, name=intg.__name__)
  else:
    intg = r['func']
  return intg


def _jit_Func(func, show_code=False, **jit_setting):
  assert isinstance(func, Function)

  r = _jit_Function(func=func, show_code=show_code, **jit_setting)
  if len(r['arguments']):
    func = _form_final_call(f_org=func._f, f_rep=r['func'], arg2call=r['arg2call'],
                            arguments=r['arguments'], nodes=r['nodes'],
                            show_code=show_code, name=func.name + '_call')
  else:
    func = r['func']

  return func


def _jit_Base(func, host, name=None, show_code=False, **jit_setting):
  r = _jit_cls_func(func, host=host, show_code=show_code, **jit_setting)
  if len(r['arguments']):
    name = func.__name__ if name is None else name
    func = _form_final_call(f_org=func, f_rep=r['func'], arg2call=r['arg2call'],
                            arguments=r['arguments'], nodes=r['nodes'],
                            show_code=show_code, name=name)
  else:
    func = r['func']
  return func


def _jit_func(obj_or_fun, show_code=False, **jit_setting):
  if callable(obj_or_fun):
    # integrator
    if isinstance(obj_or_fun, Integrator):
      return _jit_intg(obj_or_fun,
                       show_code=show_code,
                       **jit_setting)

    # bounded method
    elif hasattr(obj_or_fun, '__self__') and isinstance(obj_or_fun.__self__, Base):
      return _jit_cls_func(obj_or_fun,
                           host=obj_or_fun.__self__,
                           show_code=show_code,
                           **jit_setting)

    # wrapped function
    elif isinstance(obj_or_fun, Function):
      return _jit_Function(obj_or_fun,
                           show_code=show_code,
                           **jit_setting)

    # base class function
    elif isinstance(obj_or_fun, Base):
      return _jit_cls_func(obj_or_fun.__call__,
                           host=obj_or_fun,
                           show_code=show_code,
                           **jit_setting)

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
    _show_compiled_codes(code, code_scope)
  exec(compile(code, '', 'exec'), code_scope)
  func = code_scope[func._f.__name__]
  func = numba.jit(func, **jit_setting)

  # returns
  return dict(func=func, arguments=arguments, arg2call=arg2call, nodes=nodes)


def _jit_cls_func(f, code=None, host=None, show_code=False, **jit_setting):
  """JIT a class function.

  Examples
  --------

  Example 1: the model has static parameters.

  >>> import brainpy as bp
  >>>
  >>> class HH(bp.NeuGroup):
  >>>     def __init__(self, size, ENa=50., EK=-77., EL=-54.387, C=1.0,
  >>>                  gNa=120., gK=36., gL=0.03, V_th=20., **kwargs):
  >>>       super(HH, self).__init__(size=size, **kwargs)
  >>>       # parameters
  >>>       self.ENa = ENa
  >>>       self.EK = EK
  >>>       self.EL = EL
  >>>       self.C = C
  >>>       self.gNa = gNa
  >>>       self.gK = gK
  >>>       self.gL = gL
  >>>       self.V_th = V_th
  >>>
  >>>     def derivaitve(self, V, m, h, n, t, Iext):
  >>>       alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
  >>>       beta = 4.0 * np.exp(-(V + 65) / 18)
  >>>       dmdt = alpha * (1 - m) - beta * m
  >>>
  >>>       alpha = 0.07 * np.exp(-(V + 65) / 20.)
  >>>       beta = 1 / (1 + np.exp(-(V + 35) / 10))
  >>>       dhdt = alpha * (1 - h) - beta * h
  >>>
  >>>       alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
  >>>       beta = 0.125 * np.exp(-(V + 65) / 80)
  >>>       dndt = alpha * (1 - n) - beta * n
  >>>
  >>>       I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
  >>>       I_K = (self.gK * n ** 4.0) * (V - self.EK)
  >>>       I_leak = self.gL * (V - self.EL)
  >>>       dVdt = (- I_Na - I_K - I_leak + Iext) / self.C
  >>>
  >>>       return dVdt, dmdt, dhdt, dndt
  >>>
  >>> r = _jit_cls_func(HH(10).derivaitve, show_code=True)

  The recompiled function:
  -------------------------

  def derivaitve(V, m, h, n, t, Iext):
      alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
      beta = 4.0 * np.exp(-(V + 65) / 18)
      dmdt = alpha * (1 - m) - beta * m
      alpha = 0.07 * np.exp(-(V + 65) / 20.0)
      beta = 1 / (1 + np.exp(-(V + 35) / 10))
      dhdt = alpha * (1 - h) - beta * h
      alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
      beta = 0.125 * np.exp(-(V + 65) / 80)
      dndt = alpha * (1 - n) - beta * n
      I_Na = HH0_gNa * m ** 3.0 * h * (V - HH0_ENa)
      I_K = HH0_gK * n ** 4.0 * (V - HH0_EK)
      I_leak = HH0_gL * (V - HH0_EL)
      dVdt = (-I_Na - I_K - I_leak + Iext) / HH0_C
      return dVdt, dmdt, dhdt, dndt

  The namespace of the above function:
  {'HH0_C': 1.0,
   'HH0_EK': -77.0,
   'HH0_EL': -54.387,
   'HH0_ENa': 50.0,
   'HH0_gK': 36.0,
   'HH0_gL': 0.03,
   'HH0_gNa': 120.0,
   'bp': <module 'brainpy' from 'D:\\codes\\Projects\\BrainPy\\brainpy\\__init__.py'>}
  >>> r['func']
  CPUDispatcher(<function derivaitve at 0x0000020DF1647DC0>)
  >>> r['arguments']
  set()
  >>> r['arg2call']
  {}
  >>> r['nodes']
  {'HH0': <__main__.<locals>.HH object at 0x0000020DF1623910>}


  Example 2: the model has dynamical variables.

  >>> import brainpy as bp
  >>>
  >>> class HH(bp.NeuGroup):
  >>>     def __init__(self, size, ENa=50., EK=-77., EL=-54.387, C=1.0,
  >>>                  gNa=120., gK=36., gL=0.03, V_th=20., **kwargs):
  >>>       super(HH, self).__init__(size=size, **kwargs)
  >>>       # parameters
  >>>       self.ENa = ENa
  >>>       self.EK = EK
  >>>       self.EL = EL
  >>>       self.C = C
  >>>       self.gNa = gNa
  >>>       self.gK = gK
  >>>       self.gL = gL
  >>>       self.V_th = V_th
  >>>       self.input = bp.math.numpy.Variable(np.zeros(size))
  >>>
  >>>     def derivaitve(self, V, m, h, n, t):
  >>>       alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
  >>>       beta = 4.0 * np.exp(-(V + 65) / 18)
  >>>       dmdt = alpha * (1 - m) - beta * m
  >>>
  >>>       alpha = 0.07 * np.exp(-(V + 65) / 20.)
  >>>       beta = 1 / (1 + np.exp(-(V + 35) / 10))
  >>>       dhdt = alpha * (1 - h) - beta * h
  >>>
  >>>       alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
  >>>       beta = 0.125 * np.exp(-(V + 65) / 80)
  >>>       dndt = alpha * (1 - n) - beta * n
  >>>
  >>>       I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
  >>>       I_K = (self.gK * n ** 4.0) * (V - self.EK)
  >>>       I_leak = self.gL * (V - self.EL)
  >>>       dVdt = (- I_Na - I_K - I_leak + self.input) / self.C
  >>>
  >>>       return dVdt, dmdt, dhdt, dndt
  >>>
  >>> r = _jit_cls_func(HH(10).derivaitve, show_code=True)

  The recompiled function:
  -------------------------

  def derivaitve(V, m, h, n, t, HH0_input=None):
      alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
      beta = 4.0 * np.exp(-(V + 65) / 18)
      dmdt = alpha * (1 - m) - beta * m
      alpha = 0.07 * np.exp(-(V + 65) / 20.0)
      beta = 1 / (1 + np.exp(-(V + 35) / 10))
      dhdt = alpha * (1 - h) - beta * h
      alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
      beta = 0.125 * np.exp(-(V + 65) / 80)
      dndt = alpha * (1 - n) - beta * n
      I_Na = HH0_gNa * m ** 3.0 * h * (V - HH0_ENa)
      I_K = HH0_gK * n ** 4.0 * (V - HH0_EK)
      I_leak = HH0_gL * (V - HH0_EL)
      dVdt = (-I_Na - I_K - I_leak + HH0_input) / HH0_C
      return dVdt, dmdt, dhdt, dndt

  The namespace of the above function:
  {'HH0_C': 1.0,
   'HH0_EK': -77.0,
   'HH0_EL': -54.387,
   'HH0_ENa': 50.0,
   'HH0_gK': 36.0,
   'HH0_gL': 0.03,
   'HH0_gNa': 120.0,
   'bp': <module 'brainpy' from 'D:\\codes\\Projects\\BrainPy\\brainpy\\__init__.py'>}
  >>> r['func']
  CPUDispatcher(<function derivaitve at 0x0000020DF1647DC0>)
  >>> r['arguments']
  {'HH0_input'}
  >>> r['arg2call']
  {'HH0_input': 'HH0.input.value'}
  >>> r['nodes']
  {'HH0': <__main__.<locals>.HH object at 0x00000219AE495E80>}

  Parameters
  ----------
  f
  code
  host
  show_code
  jit_setting

  Returns
  -------

  """
  host = (host or f.__self__)

  # data to return
  arguments = set()
  arg2call = dict()
  nodes = Collector()
  nodes[host.name] = host

  # code
  code = (code or tools.deindent(inspect.getsource(f)).strip())
  # function name
  func_name = f.__name__
  # code scope
  closure_vars = inspect.getclosurevars(f)
  code_scope = dict(closure_vars.nonlocals)
  code_scope.update(closure_vars.globals)
  # analyze class function
  code, _arguments, _arg2call, _nodes, _code_scope = _analyze_cls_func(
    host=host, code=code, show_code=show_code, **jit_setting)
  arguments.update(_arguments)
  arg2call.update(_arg2call)
  nodes.update(_nodes)
  code_scope.update(_code_scope)

  # compile new function
  # code, _scope = _add_try_except(code)
  # code_scope.update(_scope)
  code_scope_to_compile = code_scope.copy()
  if show_code:
    _show_compiled_codes(code, code_scope)
  exec(compile(code, '', 'exec'), code_scope_to_compile)
  func = code_scope_to_compile[func_name]
  func = numba.jit(func, **jit_setting)

  # returns
  return dict(func=func, code=code, code_scope=code_scope,
              arguments=arguments, arg2call=arg2call, nodes=nodes)


def _jit_intg(f, show_code=False, **jit_setting):
  # TODO: integrator has "integral", "code_lines", "code_scope", "func_name", "derivative",
  assert isinstance(f, Integrator)

  # exponential euler methods
  if hasattr(f.integral, '__self__'):
    return _jit_cls_func(f=f.integral,
                         code="\n".join(f.code_lines),
                         show_code=show_code,
                         **jit_setting)

  # information in the integrator
  func_name = f.func_name
  raw_func = f.derivative
  tree = ast.parse('\n'.join(f.code_lines))
  code_scope = {key: val for key, val in f.code_scope.items()}

  # essential information
  arguments = set()
  arg2call = dict()
  nodes = Collector()

  # jit raw functions
  f_node = None
  remove_self = None
  if hasattr(f, '__self__') and isinstance(f.__self__, DynamicalSystem):
    f_node = f.__self__
    _arg = tree.body[0].args.args.pop(0)  # remove "self" arg
    # remove "self" in functional call
    remove_self = _arg.arg

  need_recompile = False
  for key, func in raw_func.items():
    # get node of host
    func_node = None
    if f_node:
      func_node = f_node
    elif hasattr(func, '__self__') and isinstance(func.__self__, DynamicalSystem):
      func_node = func.__self__

    # get new compiled function
    if isinstance(func, Dispatcher):
      continue
    elif func_node is not None:
      need_recompile = True
      r = _jit_cls_func(f=func,
                        host=func_node,
                        show_code=show_code,
                        **jit_setting)
      if len(r['arguments']) or remove_self:
        tree = _replace_func_call_by_tree(tree,
                                          func_call=key,
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
    # code_scope_backup = {k: v for k, v in code_scope.items()}
    # compile functions
    if show_code:
      _show_compiled_codes(code, code_scope)
    exec(compile(code, '', 'exec'), code_scope)
    new_f = code_scope[func_name]
    # new_f.brainpy_data = {key: val for key, val in f.brainpy_data.items()}
    # new_f.brainpy_data['code_lines'] = code.strip().split('\n')
    # new_f.brainpy_data['code_scope'] = code_scope_backup
    jit_f = numba.jit(new_f, **jit_setting)
    return dict(func=jit_f, arguments=arguments, arg2call=arg2call, nodes=nodes)
  else:
    return dict(func=f, arguments=arguments, arg2call=arg2call, nodes=nodes)


def _analyze_cls_func(host, code, show_code, self_name=None, pop_self=True, **jit_setting):
  """Analyze the bounded function of one object.

  Parameters
  ----------
  host : Base
    The data host.
  code : str
    The function source code.
  self_name : optional, str
    The class name, like "self", "cls".
  show_code : bool
  """
  # arguments
  tree = ast.parse(code)
  if self_name is None:
    self_name = tree.body[0].args.args[0].arg
    # data assigned by self.xx in line right
    if self_name not in CLASS_KEYWORDS:
      raise errors.CodeError(f'BrainPy only support class keyword '
                             f'{CLASS_KEYWORDS}, but we got {self_name}.')
  if pop_self:
    tree.body[0].args.args.pop(0)  # remove "self" etc. class argument

  # analyze function body
  r = _analyze_cls_func_body(host=host,
                             self_name=self_name,
                             code=code,
                             tree=tree,
                             show_code=show_code,
                             has_func_def=True,
                             **jit_setting)
  code, arguments, arg2call, nodes, code_scope = r

  return code, arguments, arg2call, nodes, code_scope


def _analyze_cls_func_body(host,
                           self_name,
                           code,
                           tree,
                           show_code=False,
                           has_func_def=False,
                           **jit_setting):
  arguments, arg2call, nodes, code_scope = set(), dict(), Collector(), dict()

  # all self data
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
      if isinstance(next_target, Integrator):
        break
      if not isinstance(next_target, Base):
        break
      target = next_target
    else:
      raise errors.BrainPyError
    data = getattr(target, split_keys[i])

    # analyze data
    if isinstance(data, math.numpy.Variable):  # data is a variable
      arguments.add(f'{target.name}_{split_keys[i]}')
      arg2call[f'{target.name}_{split_keys[i]}'] = f'{target.name}.{split_keys[-1]}.value'
      nodes[target.name] = target
      # replace the data
      if len(split_keys) == i + 1:
        data_to_replace[key] = f'{target.name}_{split_keys[i]}'
      else:
        data_to_replace[key] = f'{target.name}_{split_keys[i]}.{".".join(split_keys[i + 1:])}'

    elif isinstance(data, np.random.RandomState):  # data is a RandomState
      # replace RandomState
      code_scope[f'{target.name}_{split_keys[i]}'] = np.random
      # replace the data
      if len(split_keys) == i + 1:
        data_to_replace[key] = f'{target.name}_{split_keys[i]}'
      else:
        data_to_replace[key] = f'{target.name}_{split_keys[i]}.{".".join(split_keys[i + 1:])}'

    elif callable(data):  # data is a function
      assert len(split_keys) == i + 1
      r = _jit_func(obj_or_fun=data, show_code=show_code, **jit_setting)
      # if len(r['arguments']):
      tree = _replace_func_call_by_tree(tree, func_call=key, arg_to_append=r['arguments'])
      arguments.update(r['arguments'])
      arg2call.update(r['arg2call'])
      nodes.update(r['nodes'])
      code_scope[f'{target.name}_{split_keys[i]}'] = r['func']
      data_to_replace[key] = f'{target.name}_{split_keys[i]}'  # replace the data

    elif isinstance(data, (dict, list, tuple)):  # data is a list/tuple/dict of function/object
      # get all values
      if isinstance(data, dict):  # check dict
        if len(split_keys) != i + 2 and split_keys[-1] != 'values':
          raise errors.BrainPyError(f'Only support iter dict.values(). while we got '
                                    f'dict.{split_keys[-1]}  for data: \n\n{data}')
        values = list(data.values())
        iter_name = key + '()'
      else:  # check list / tuple
        assert len(split_keys) == i + 1
        values = list(data)
        iter_name = key
        if len(values) > 0:
          if not (callable(values[0]) or isinstance(values[0], Base)):
            code_scope[f'{target.name}_{split_keys[i]}'] = data
            if len(split_keys) == i + 1:
              data_to_replace[key] = f'{target.name}_{split_keys[i]}'
            else:
              data_to_replace[key] = f'{target.name}_{split_keys[i]}.{".".join(split_keys[i + 1:])}'
            continue
            # raise errors.BrainPyError(f'Only support JIT an iterable objects of function '
            #                           f'or Base object, but we got:\n\n {values[0]}')
      # replace this for-loop
      r = _replace_this_forloop(tree=tree,
                                iter_name=iter_name,
                                loop_values=values,
                                show_code=show_code,
                                **jit_setting)
      tree, _arguments, _arg2call, _nodes, _code_scope = r
      arguments.update(_arguments)
      arg2call.update(_arg2call)
      nodes.update(_nodes)
      code_scope.update(_code_scope)

    else:  # constants
      code_scope[f'{target.name}_{split_keys[i]}'] = data
      # replace the data
      if len(split_keys) == i + 1:
        data_to_replace[key] = f'{target.name}_{split_keys[i]}'
      else:
        data_to_replace[key] = f'{target.name}_{split_keys[i]}.{".".join(split_keys[i + 1:])}'

  if has_func_def:
    tree.body[0].decorator_list.clear()
    tree.body[0].args.args.extend([ast.Name(id=a) for a in sorted(arguments)])
    tree.body[0].args.defaults.extend([ast.Constant(None) for _ in sorted(arguments)])
    tree.body[0].args.kwarg = None

  # replace words
  code = tools.ast2code(tree)
  code = tools.word_replace(code, data_to_replace, exclude_dot=True)

  return code, arguments, arg2call, nodes, code_scope


def _replace_this_forloop(tree, iter_name, loop_values, show_code=False, **jit_setting):
  """Replace the given for-loop.

  This function aims to replace the specific for-loop structure, like:

  replace this for-loop

  >>> def update(_t, _dt):
  >>>    for step in self.child_steps.values():
  >>>        step(_t, _dt)

  to

  >>> def update(_t, _dt, AMPA_vec0_delay_g_data=None, AMPA_vec0_delay_g_in_idx=None,
  >>>            AMPA_vec0_delay_g_out_idx=None, AMPA_vec0_s=None, HH0_V=None, HH0_V_th=None,
  >>>            HH0_gNa=None, HH0_h=None, HH0_input=None, HH0_m=None, HH0_n=None, HH0_spike=None):
  >>>    HH0_step(_t, _dt, HH0_V=HH0_V, HH0_V_th=HH0_V_th, HH0_gNa=HH0_gNa,
  >>>             HH0_h=HH0_h, HH0_input=HH0_input, HH0_m=HH0_m, HH0_n=HH0_n,
  >>>             HH0_spike=HH0_spike)
  >>>    AMPA_vec0_step(_t, _dt, AMPA_vec0_delay_g_data=AMPA_vec0_delay_g_data,
  >>>                   AMPA_vec0_delay_g_in_idx=AMPA_vec0_delay_g_in_idx,
  >>>                   AMPA_vec0_delay_g_out_idx=AMPA_vec0_delay_g_out_idx,
  >>>                   AMPA_vec0_s=AMPA_vec0_s, HH0_V=HH0_V, HH0_input=HH0_input,
  >>>                   HH0_spike=HH0_spike)
  >>>    AMPA_vec0_delay_g_step(_t, _dt, AMPA_vec0_delay_g_in_idx=AMPA_vec0_delay_g_in_idx,
  >>>                           AMPA_vec0_delay_g_out_idx=AMPA_vec0_delay_g_out_idx)

  Parameters
  ----------
  tree : ast.Module
    The target code tree.
  iter_name : str
    The for-loop iter.
  loop_values : list/tuple
    The iter contents in the current loop.
  show_code : bool
    Whether show the formatted code.
  """
  assert isinstance(tree, ast.Module)

  replacer = ReplaceThisForLoop(loop_values=loop_values,
                                iter_name=iter_name,
                                show_code=show_code,
                                **jit_setting)
  tree = replacer.visit(tree)
  if not replacer.success:
    raise errors.BrainPyError(f'Do not find the for-loop for "{iter_name}", '
                              f'currently we only support for-loop like '
                              f'"for xxx in {iter_name}:". Does your for-loop '
                              f'structure is not like this. ')

  return tree, replacer.arguments, replacer.arg2call, replacer.nodes, replacer.code_scope


class ReplaceThisForLoop(ast.NodeTransformer):
  def __init__(self, loop_values, iter_name, show_code=False, **jit_setting):
    self.success = False

    # targets
    self.loop_values = loop_values
    self.iter_name = iter_name

    # setting
    self.show_code = show_code
    self.jit_setting = jit_setting

    # results
    self.arguments = set()
    self.arg2call = dict()
    self.nodes = Collector()
    self.code_scope = dict()

  def visit_For(self, node):
    iter_ = tools.ast2code(ast.fix_missing_locations(node.iter))

    if iter_.strip() == self.iter_name:
      data_to_replace = Collector()
      final_node = ast.Module(body=[])
      self.success = True

      # target
      if not isinstance(node.target, ast.Name):
        raise errors.BrainPyError(f'Only support scalar iter, like "for x in xxxx:", not "for '
                                  f'{tools.ast2code(ast.fix_missing_locations(node.target))} '
                                  f'in {iter_}:')
      target = node.target.id

      # for loop values
      for i, value in enumerate(self.loop_values):
        # module and code
        module = ast.Module(body=deepcopy(node).body)
        code = tools.ast2code(module)

        if isinstance(value, Base):  # transform Base objects
          r = _analyze_cls_func_body(host=value,
                                     self_name=target,
                                     code=code,
                                     tree=module,
                                     show_code=self.show_code,
                                     **self.jit_setting)

          new_code, arguments, arg2call, nodes, code_scope = r
          self.arguments.update(arguments)
          self.arg2call.update(arg2call)
          self.arg2call.update(arg2call)
          self.nodes.update(nodes)
          self.code_scope.update(code_scope)

          final_node.body.extend(ast.parse(new_code).body)

        elif callable(value):  # transform functions
          r = _jit_func(obj_or_fun=value,
                        show_code=self.show_code,
                        **self.jit_setting)
          tree = _replace_func_call_by_tree(deepcopy(module),
                                            func_call=target,
                                            arg_to_append=r['arguments'],
                                            new_func_name=f'{target}_{i}')

          # update import parameters
          self.arguments.update(r['arguments'])
          self.arg2call.update(r['arg2call'])
          self.nodes.update(r['nodes'])

          # replace the data
          if isinstance(value, Base):
            host = value
            replace_name = f'{host.name}_{target}'
          elif hasattr(value, '__self__') and isinstance(value.__self__, Base):
            host = value.__self__
            replace_name = f'{host.name}_{target}'
          else:
            replace_name = f'{target}_{i}'
          self.code_scope[replace_name] = r['func']
          data_to_replace[f'{target}_{i}'] = replace_name

          final_node.body.extend(tree.body)

        else:
          raise errors.BrainPyError(f'Only support JIT an iterable objects of function '
                                    f'or Base object, but we got:\n\n {value}')

      # replace words
      final_code = tools.ast2code(final_node)
      final_code = tools.word_replace(final_code, data_to_replace, exclude_dot=True)
      final_node = ast.parse(final_code)

    else:
      final_node = node

    self.generic_visit(final_node)
    return final_node


def _replace_func_call_by_tree(tree, func_call, arg_to_append, remove_self=None,
                               new_func_name=None):
  assert isinstance(func_call, str)
  assert isinstance(arg_to_append, (list, tuple, set))
  assert isinstance(tree, ast.Module)

  transformer = FuncTransformer(func_name=func_call,
                                arg_to_append=arg_to_append,
                                remove_self=remove_self,
                                new_func_name=new_func_name)
  new_tree = transformer.visit(tree)
  return new_tree


def _replace_func_call_by_code(code, func_call, arg_to_append, remove_self=None):
  """Replace functional call.

  This class automatically transform a functional call.
  For example, in your original code:

  >>> V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input)

  you want to add new arguments ``gNa``, ``gK`` and ``gL`` into the
  function ``self.integral``. Then this Transformer will help you
  automatically do this:

  >>> V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input,
  >>>                            gNa=gNa, gK=gK, gL=gL)

  Parameters
  ----------
  code : str
    The original code string.
  func_call : str
    The functional call.
  arg_to_append : set/list/tuple of str
    The arguments to append.
  remove_self : str, optional
    The self class name to remove.

  Returns
  -------
  new_code : str
    The new code string.
  """
  assert isinstance(func_call, str)
  assert isinstance(arg_to_append, (list, tuple, set))

  tree = ast.parse(code)
  transformer = FuncTransformer(func_name=func_call,
                                arg_to_append=arg_to_append,
                                remove_self=remove_self)
  new_tree = transformer.visit(tree)
  return tools.ast2code(new_tree)


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

  def __init__(self, func_name, arg_to_append, remove_self=None, new_func_name=None):
    self.func_name = func_name
    self.new_func_name = new_func_name
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
      arg_to_append = deepcopy(self.arg_to_append)
      for arg in kwargs:
        if arg.arg in arg_to_append:
          arg_to_append.remove(arg.arg)
      if len(arg_to_append):
        code = f'f({", ".join([f"{k}={k}" for k in arg_to_append])})'
        tree = ast.parse(code)
        new_keywords = tree.body[0].value.keywords
        kwargs.extend(new_keywords)
      # final function
      if self.new_func_name:
        func_call = ast.parse(f'{self.new_func_name}()').body[0].value.func
      else:
        func_call = node.func
      return ast.Call(func=func_call, args=args, keywords=kwargs)
    return node


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


def _items2lines(items, num_each_line=1, separator=', ', line_break='\n\t\t'):
  res = ''
  for item in items[:num_each_line]:
    res += item + separator
  for i in range(num_each_line, len(items), num_each_line):
    res += line_break
    for item in items[i: i + num_each_line]:
      res += item + separator
  return res


def _get_args(f):
  # 1. get the function arguments
  original_args = []
  args = []
  kwargs = []

  for name, par in inspect.signature(f).parameters.items():
    if par.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
      args.append(par.name)
    elif par.kind is inspect.Parameter.VAR_POSITIONAL:
      args.append(par.name)
    elif par.kind is inspect.Parameter.KEYWORD_ONLY:
      args.append(par.name)
    elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
      raise errors.BrainPyError('Don not support positional only parameters, e.g., /')
    elif par.kind is inspect.Parameter.VAR_KEYWORD:
      kwargs.append(par.name)
    else:
      raise errors.BrainPyError(f'Unknown argument type: {par.kind}')

    original_args.append(str(par))

  # 2. analyze the function arguments
  #   2.1 class keywords
  class_kw = []
  if original_args[0] in CLASS_KEYWORDS:
    class_kw.append(original_args[0])
    original_args = original_args[1:]
    args = args[1:]
  for a in original_args:
    if a.split('=')[0].strip() in CLASS_KEYWORDS:
      raise errors.DiffEqError(f'Class keywords "{a}" must be defined '
                               f'as the first argument.')
  return class_kw, args, kwargs, original_args


def _form_final_call(f_org, f_rep, arg2call, arguments, nodes, show_code=False, name=None):
  _, args, kwargs, org_args = _get_args(f_org)

  name = (name or f_org.__name__)
  code_scope = {key: node for key, node in nodes.items()}
  code_scope[name] = f_rep

  new_args = [] + args
  new_args += [f"{a}={arg2call[a]}" for a in sorted(arguments)]
  new_args += [f"**{a}" for a in kwargs]
  called_args = _items2lines(new_args).strip()
  code_lines = [f'def new_{name}({", ".join(org_args)}):',
                f'  {name}({called_args.strip()})']

  # compile new function
  code = '\n'.join(code_lines)
  # code, _scope = _add_try_except(code)
  # code_scope.update(_scope)
  if show_code:
    _show_compiled_codes(code, code_scope)
  exec(compile(code, '', 'exec'), code_scope)
  func = code_scope[f'new_{name}']
  return func


def _show_compiled_codes(code, scope):
  print('The recompiled function:')
  print('-------------------------')
  print(code)
  print()
  print('The namespace of the above function:')
  pprint(scope)
  print()


def _find_all_forloop(code_or_tree):
  """Find all for-loops in the code.

  >>> code = '''
  >>> for ch in self._update_channels:
  >>>  ch.update(_t, _dt)
  >>> for ch in self._current_channels:
  >>>  self.input += ch.update(_t, _dt)
  >>> '''
  >>> _find_all_forloop(code)
  {'self._current_channels': ('ch',
                            <_ast.Module object at 0x00000155BD23B730>,
                            'self.input += ch.update(_t, _dt)\n'),
  'self._update_channels': ('ch',
                            <_ast.Module object at 0x00000155B699AD90>,
                            'ch.update(_t, _dt)\n')}

  >>> code = '''
  >>> self.pre_spike.push(self.pre.spike)
  >>> pre_spike = self.pre_spike.pull()
  >>>
  >>> self.g[:] = self.integral(self.g, _t, dt=_dt)
  >>> for pre_id in range(self.pre.num):
  >>>   if pre_spike[pre_id]:
  >>>     start, end = self.pre_slice[pre_id]
  >>>     for post_id in self.post_ids[start: end]:
  >>>       self.g[post_id] += self.g_max
  >>>
  >>> self.post.input[:] += self.output_current(self.g)
  >>>   '''
  >>> _find_all_forloop(code)
  {'range(self.pre.num)': ('pre_id',
                         <_ast.Module object at 0x000001D0AB120D60>,
                         'if pre_spike[pre_id]:\n'
                         '    start, end = self.pre_slice[pre_id]\n'
                         '    for post_id in self.post_ids[start:end]:\n'
                         '        self.g[post_id] += self.g_max\n'),
  'self.post_ids[start:end]': ('post_id',
                              <_ast.Module object at 0x000001D0AB11B460>,
                              'self.g[post_id] += self.g_max\n')}

  Parameters
  ----------
  code_or_tree: str, ast.Module

  Returns
  -------
  res : dict
    with <iter, (target, body, code)>
  """

  # code or tree
  if isinstance(code_or_tree, str):
    code_or_tree = ast.parse(code_or_tree)
  elif isinstance(code_or_tree, ast.Module):
    code_or_tree = code_or_tree
  else:
    raise ValueError

  # finder
  finder = FindAllForLoop()
  finder.visit(code_or_tree)

  # dictionary results
  res = dict()
  for iter_, target, body, body_str in zip(finder.for_iter, finder.for_target,
                                           finder.for_body, finder.for_body_str):
    res[iter_] = (target, body, body_str)
  return res


class FindAllForLoop(ast.NodeVisitor):
  def __init__(self):
    self.for_iter = []
    self.for_target = []
    self.for_body = []
    self.for_body_str = []

  def visit_For(self, node):
    self.for_target.append(tools.ast2code(ast.fix_missing_locations(node.target)))
    self.for_iter.append(tools.ast2code(ast.fix_missing_locations(node.iter)))
    self.for_body.append(ast.Module(body=[deepcopy(r) for r in node.body]))
    codes = tuple(tools.ast2code(ast.fix_missing_locations(r)) for r in node.body)
    self.for_body_str.append('\n'.join(codes))

    self.generic_visit(node)
