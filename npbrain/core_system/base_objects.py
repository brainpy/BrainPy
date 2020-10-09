# -*- coding: utf-8 -*-

import inspect
import re
from copy import deepcopy
from importlib import import_module
from pprint import pprint

import autopep8

from .types import NeuState
from .types import ObjState
from .types import TypeChecker
from .types import TypeMismatchError
from .types import _SynStateForNbSingleMode
from .. import _numpy as np
from .. import profile
from .. import tools
from ..integration import Integrator
from ..integration.sympy_tools import get_mapping_scope

__all__ = [
    # errors
    'ModelDefError',

    # base types
    'BaseType',
    'BaseEnsemble',
]

_ARG_KEYWORDS = ['_dt_', '_t_', '_i_']
_NEU_GROUP = 'NeuGroup'
_SYN_CONN = 'SynConn'
_NEU_TYPE = 'NeuType'
_SYN_TYPE = 'SynType'
_NEU_NO = 0
_SYN_NO = 0


class ModelDefError(Exception):
    """Model definition error."""
    pass


class ModelUseError(Exception):
    """Model use error."""
    pass


class BaseType(object):
    """The base type of neuron and synapse.

    Structure of a BaseType instantiation:

    - parameters = a_dict  (model parameters)
    - variables = a_dict   (model variables)
    - attributes = a_dict  (essential attributes for model running)
     - a1 = av1
     - a2 = av2
     - ...
    - steps = a_list  (collection of the step functions)
     - f1 = callable
     - f2 = callable
     - ...

    Parameters
    ----------
    create_func : callable
        The function to create the model.
    name : str, optional
        Model name.
    vector_based : bool
        Whether the model is written in the neuron-group level or in the single-neuron level.
    type_ : str
        Whether the model is a 'neuron' or a 'synapse' model.
    """

    def __init__(self, create_func, name=None, vector_based=True, type_=_NEU_TYPE):
        # type : neuron based or group based code
        # ---------------------------------------
        self.vector_based = vector_based

        # name
        # -----
        if name is None:
            if type_ == _NEU_TYPE:
                global _NEU_NO
                self.name = f'NeuType{_NEU_NO}'
                _NEU_NO += 1
            elif type_ == _SYN_TYPE:
                global _SYN_NO
                self.name = f'SynType{_SYN_NO}'
                _SYN_NO += 1
            else:
                raise ModelDefError(f'Unknown model type "{type_}", only support "{_NEU_TYPE}" and "{_SYN_TYPE}".')
        else:
            self.name = name

        # create_func
        # ------------
        try:
            func_return = create_func()
        except TypeError as e:
            raise ModelDefError(f'Arguments in "{create_func.__name__}" must provide default values.')
        if not isinstance(func_return, dict):
            raise ModelDefError('"create_func" must return a dict.')
        try:
            assert 'requires' in func_return
        except AssertionError:
            raise ModelDefError('"requires" (specify variables the model need) must be defined in the return.')
        if vector_based:
            try:
                assert 'steps' in func_return
            except AssertionError:
                raise ModelDefError('"steps" (step functions at each time step) must be defined in the return.')
        else:
            try:
                assert 'update' in func_return
            except AssertionError:
                raise ModelDefError('"update" function must be defined in the return.')
            if type_ == _SYN_TYPE:
                try:
                    assert 'output' in func_return
                except AssertionError:
                    raise ModelDefError('"output" function must be defined in the return.')

        self.create_func = create_func
        self.func_return = func_return

        # parameters
        # -----------
        parameters = inspect.getcallargs(create_func)
        self.parameters = dict(parameters)

        # attributes
        # -----------
        requires = func_return['requires']
        try:
            assert isinstance(requires, dict)
        except AssertionError:
            raise ModelDefError('"requires" only supports dict.')
        try:
            assert 'ST' in requires
        except AssertionError:
            raise ModelDefError('"ST" must be defined in "requires".')
        self.requires = requires
        for k, v in requires.items():
            if isinstance(v, type):
                raise ModelDefError(f'In "requires", you must instantiate the type checker of "{k}". '
                                    f'Like "{v.__name__}()".')
            try:
                assert isinstance(v, TypeChecker)
            except AssertionError:
                raise ModelDefError(f'In "requires", each value must be a {TypeChecker.__name__}, '
                                    f'but got "{type(v)}" for "{k}".')
            setattr(self, k, v)

        # variables
        # ----------
        self.variables = self.requires['ST']._vars

        # step functions
        # --------------
        self.steps, self.step_names = [], []
        if vector_based:
            steps = func_return['steps']
            if callable(steps):
                steps = [steps]
            elif isinstance(steps, (list, tuple)):
                steps = list(steps)
            else:
                raise ModelDefError('"steps" must be a callable, or a list/tuple of callable functions.')
            for func in steps:
                try:
                    assert callable(func)
                except AssertionError:
                    raise ModelDefError('"steps" must be a list/tuple of callable functions.')
                func_name = func.__name__
                self.step_names.append(func_name)
                self.steps.append(func)
                setattr(self, func_name, func)
        else:
            update = func_return['update']
            self.steps.append(update)
            self.step_names.append('update')
            setattr(self, 'update', update)
            if type_ == _SYN_TYPE:
                output = func_return['output']
                self.steps.append(output)
                self.step_names.append('output')
                setattr(self, 'output', output)

        # check consistence between function
        # arguments and model attributes
        # ----------------------------------
        warnings = []
        for func in self.steps:
            for arg in inspect.getfullargspec(func).args:
                if arg in _ARG_KEYWORDS:
                    continue
                if arg not in self.requires:
                    warn = f'"{self.name}" requires "{arg}" as argument, but "{arg}" isn\'t declared in "requires".'
                    warnings.append(warn)
        print('\n'.join(warnings) + '\n')

    def __str__(self):
        return f'{self.name}'


class BaseEnsemble(object):
    """Base Ensemble class.

    Parameters
    ----------
    model : BaseType
        The (neuron/synapse) model type.

    """

    def __init__(self, model, name, num, pars_update, vars_init, monitors, cls_type):
        # class type
        # -----------
        assert cls_type in [_NEU_GROUP, _SYN_CONN], f'Only support "{_NEU_GROUP}" and "{_SYN_CONN}".'
        self._cls_type = cls_type

        # model
        # -----
        self.model = model

        # name
        # ----
        self.name = name
        if not self.name.isidentifier():
            raise ValueError(f'"{self.name}" isn\'t a valid identifier according to Python '
                             f'language definition. Please choose another name.')

        # num
        # ---
        self.num = num

        # variables
        # ---------
        vars_init = dict() if vars_init is None else vars_init
        try:
            assert isinstance(vars_init, dict)
        except AssertionError:
            raise ModelUseError('"vars_init" must be a dict.')
        variables = deepcopy(self.model.variables)
        for k, v in vars_init:
            if k not in self.model.variables:
                raise ModelUseError(f'variable "{k}" is not defined in "{self.model.name}".')
            variables[k] = v
        self.vars_init = variables

        # parameters
        # ----------
        self._hetero_pars = {}
        pars_update = dict() if pars_update is None else pars_update
        assert isinstance(pars_update, dict), '"pars_update" must be a dict.'
        parameters = deepcopy(self.model.parameters)
        for k, v in pars_update.items():
            val_size = np.size(v)
            if val_size != 1:
                if val_size != num:
                    raise ModelUseError(f'The size of parameter "{k}" is wrong, "{val_size}" != 1 '
                                        f'and "{val_size}" != {num}.')
                else:
                    self._hetero_pars[k] = v
            parameters[k] = v
        self.pars_update = parameters

        # step functions
        # ---------------
        if not model.vector_based:
            if self._cls_type == _SYN_CONN and (self.pre_group is None or self.post_group is None):
                raise ModelUseError('Single synapse model must provide "pre_group" and "post_group".')
        self._steps = self._get_steps_from_model(self.pars_update)

        # monitors
        # ---------
        self.mon = tools.DictPlus()
        self._mon_vars = []
        if monitors is not None:
            if isinstance(monitors, (list, tuple)):
                for var in monitors:
                    if isinstance(var, str):
                        self._mon_vars.append((var, None))
                        self.mon[var] = np.empty((1, 1), dtype=np.float_)
                    elif isinstance(var, (tuple, list)):
                        self._mon_vars.append((var[0], var[1]))
                        self.mon[var[0]] = np.empty((1, 1), dtype=np.float_)
                    else:
                        raise ModelUseError(f'Unknown monitor item: {str(var)}')
            elif isinstance(monitors, dict):
                for k, v in monitors.items():
                    self._mon_vars.append((k, v))
                    self.mon[k] = np.empty((1, 1), dtype=np.float_)
            else:
                raise ModelUseError(f'Unknown monitors type: {type(monitors)}')

        # code generation results
        # -----------------------
        self._codegen = dict()

        # model update schedule
        # ---------------------
        self._schedule = ['input'] + self.model.step_names + ['monitor']

    def _get_steps_from_model(self, pars_update):
        if self.model.vector_based:
            func_return = self.model.create_func(**pars_update)
            steps = func_return['steps']
            if callable(steps):
                steps = [steps, ]
            elif isinstance(steps, (tuple, list)):
                steps = list(steps)
            else:
                raise ModelDefError('"steps" must be a callable, or a list/tuple of callable functions.')
        else:
            steps = []
            func_return = self.model.create_func(**pars_update)
            steps.append(func_return['update'])
            if self._cls_type == _SYN_CONN:
                steps.append(func_return['output'])
        return steps

    def _type_checking(self):
        # check attribute and its type
        for key, type_checker in self.model.requires.items():
            if not hasattr(self, key):
                raise ModelUseError(f'"{self.name}" doesn\'t have "{key}" attribute.')
            try:
                type_checker.check(getattr(self, key))
            except TypeMismatchError as e:
                raise ModelUseError(f'"{self.name}.{key}" doesn\'t satisfy TypeChecker "{str(type_checker)}".')

        # get function arguments
        for i, func in enumerate(self._steps):
            for arg in inspect.getfullargspec(func).args:
                if not (arg in _ARG_KEYWORDS + ['self']) and not hasattr(self, arg):
                    raise ModelUseError(f'Function "{self._steps[i].__name__}" in "{self.model.name}" requires '
                                        f'"{arg}" as argument, but "{arg}" is not defined in "{self.name}".')

    def __step_substitute_integrator(self, func):
        func_code = tools.deindent(tools.get_main_code(func))
        code_lines = tools.get_code_lines(func_code)

        # get function scope
        vars = inspect.getclosurevars(func)
        code_scope = dict(vars.nonlocals)
        code_scope.update(vars.globals)
        code_scope.update({self.name: self})
        if len(code_lines) == 0:
            return '', code_scope

        scope_to_add = {}
        scope_to_del = set()
        need_add_mapping_scope = False
        for k, v in code_scope.items():
            if isinstance(v, Integrator):
                need_add_mapping_scope = True

                # locate the integration function
                int_func_name = v.py_func_name
                for line_no, line in enumerate(code_lines):
                    if int_func_name in tools.get_identifiers(line):
                        break

                # get integral function line indent
                line_indent = tools.get_line_indent(line)
                indent = ' ' * line_indent

                # get the replace line and arguments need to replace
                new_line, args, kwargs = tools.func_replace(line, int_func_name)

                # append code line of argument replacement
                func_args = v.diff_eqs.func_args
                append_lines = [indent + f'_{v.py_func_name}_{func_args[i]} = {args[i]}'
                                for i in range(len(args))]
                for arg in func_args[len(args):]:
                    append_lines.append(indent + f'_{v.py_func_name}_{arg} = {kwargs[arg]}')

                # append numerical integration code lines
                append_lines.extend([indent + l for l in v.update_code.split('\n')])
                append_lines.append(indent + new_line)

                # add appended lines into the main function code lines
                code_lines = code_lines[:line_no] + append_lines + code_lines[line_no + 1:]

                # get scope variables to delete
                scope_to_del.add(k)
                for k2, v2 in v.code_scope.items():
                    if callable(v2):
                        v2 = tools.numba_func(v2)
                    scope_to_add[k2] = v2

        # update code scope
        if need_add_mapping_scope:
            code_scope.update(get_mapping_scope())
        code_scope.update(scope_to_add)
        for k in scope_to_del:
            code_scope.pop(k)

        # return code lines and code scope
        return '\n'.join(code_lines), code_scope

    def __step_mode_np_group(self):
        for func in self._steps:
            func_name = func.__name__
            func_args = inspect.getfullargspec(func).args

            setattr(self, func_name, func)
            arg_calls = []
            for arg in func_args:
                if arg in _ARG_KEYWORDS:
                    arg_calls.append(arg)
                else:
                    arg_calls.append(f"{self.name}.{arg}")
            func_call = f'{self.name}.{func_name}({", ".join(arg_calls)})'
            self._codegen[func_name] = {'func': func, 'call': func_call}

    def __step_mode_np_single(self):
        if self.num > 1000:
            raise ModelUseError(f'The number of the '
                                f'{"neurons" if self._cls_type == _NEU_GROUP else "synapses"} is '
                                f'too huge (>1000), please use numba backend or define vector_based model.')

        # get step functions
        steps_collection = {func.__name__: [] for func in self._steps}
        for i in range(self.num):
            pars = {k: v if k not in self._hetero_pars else self._hetero_pars[k][i]
                    for k, v in self.pars_update.items()}
            steps = self._get_steps_from_model(pars)
            for func in steps:
                steps_collection[func.__name__].append(func)

        for func in self._steps:
            func_name = func.__name__
            func_args = inspect.getfullargspec(func).args
            state_args = [arg for arg in func_args
                          if arg not in _ARG_KEYWORDS and
                          isinstance(getattr(self, arg), ObjState)]

            # arg and arg2call
            code_arg, code_arg2call = [], {}
            for arg in func_args:
                if arg in state_args:
                    arg2 = f'{self.name}_{arg}'
                    code_arg2call[arg2] = f'{self.name}.{arg}'
                    code_arg.append(arg2)
                else:
                    if arg in _ARG_KEYWORDS:
                        code_arg2call[arg] = arg
                    else:
                        code_arg2call[arg] = f'{self.name}.{arg}'
                    code_arg.append(arg)

            # scope
            code_scope = {f'{func_name}_collect': steps_collection[func_name]}

            # codes
            has_ST = 'ST' in state_args
            has_pre = 'pre' in state_args
            has_post = 'post' in state_args
            if has_ST:
                if has_pre and has_post:
                    code_arg.extend(['pre2syn', 'post_idx'])
                    code_arg2call['pre2syn'] = f'{self.name}.pre2syn'
                    code_arg2call['post_idx'] = f'{self.name}.post_idx'

                    code_lines = [f'\n# "{func_name}" step function in {self.name}\n'
                                  f'def {func_name}({", ".join(code_arg)}):',
                                  f'for pre_idx in range({self.pre_group.num}):',
                                  f'  pre = {self.name}_pre.extract_by_index(pre_idx)',
                                  f'  for syn_idx in pre2syn[pre_idx]:',
                                  f'    post_i = post_idx[syn_idx]',
                                  f'    post = {self.name}_post.extract_by_index(post_i)',
                                  f'    ST = {self.name}_ST.extract_by_index(syn_idx)',
                                  f'    {func_name}_collect[syn_idx]({", ".join(func_args)})',
                                  f'    {self.name}_ST.update_by_index(syn_idx, ST)']

                elif has_pre:
                    code_arg.append('pre2syn')
                    code_arg2call['pre2syn'] = f'{self.name}.pre2syn'

                    code_lines = [f'\n# "{func_name}" step function in {self.name}\n'
                                  f'def {func_name}({", ".join(code_arg)}):',
                                  f'for pre_idx in range({self.pre_group.num}):',
                                  f'  pre = {self.name}_pre.extract_by_index(pre_idx)',
                                  f'  for syn_idx in pre2syn[pre_idx]:',
                                  f'    ST = {self.name}_ST.extract_by_index(syn_idx)',
                                  f'    {func_name}_collect[syn_idx]({", ".join(func_args)})',
                                  f'    {self.name}_ST.update_by_index(syn_idx, ST)']

                elif has_post:
                    code_arg.append('post2syn')
                    code_arg2call['post2syn'] = f'{self.name}.post2syn'

                    code_lines = [f'\n# "{func_name}" step function in {self.name}\n'
                                  f'def {func_name}({", ".join(code_arg)}):',
                                  f'for post_idx in range({self.post_group.num}):',
                                  f'  post = {self.name}_post.extract_by_index(post_idx)',
                                  f'  for syn_idx in post2syn[post_idx]:',
                                  f'    ST = {self.name}_ST.extract_by_index(syn_idx)',
                                  f'    {func_name}_collect[syn_idx]({", ".join(func_args)})',
                                  f'    {self.name}_ST.update_by_index(syn_idx, ST)']

                else:
                    code_lines = [f'\n# "{func_name}" step function in {self.name}\n'
                                  f'def {func_name}({", ".join(code_arg)}):',
                                  f'for _ni_ in range({self.num}):']
                    for arg in state_args:
                        code_lines.append(f'  {arg} = {self.name}_{arg}.extract_by_index(_ni_)')
                    code_lines.append(f'  {func_name}_collect[_ni_]({", ".join(func_args)})')
                    for arg in state_args:
                        code_lines.append(f'  {self.name}_{arg}.update_by_index(_ni_, {arg})')

            else:
                try:
                    assert not has_post and not has_pre
                except AssertionError:
                    raise ModelDefError(f'Unknown "{func_name}" function structure.')
                code_lines = [f'\n# "{func_name}" step function in {self.name}\n'
                              f'def {func_name}({", ".join(code_arg)}):',
                              f'for _ni_ in range({self.num}):',
                              f'  {func_name}_collect[_ni_]({", ".join(func_args)})']

            # compile
            func_code = '\n  '.join(code_lines)
            if profile.auto_pep8:
                func_code = autopep8.fix_code(func_code)
            exec(compile(func_code, '', 'exec'), code_scope)
            func = code_scope[func_name]
            setattr(self, func_name, func)

            # call
            func_call = f'{self.name}.{func_name}({", ".join([code_arg2call[arg] for arg in code_arg])})'

            if profile.show_codgen:
                print(func_code)
                print(func_call)

            # final
            self._codegen[func_name] = {'func': func, 'call': func_call}

    def __step_mode_nb_group(self):
        for func in self._steps:
            func_name = func.__name__
            func_args = inspect.getfullargspec(func).args
            states = {k: getattr(self, k) for k in func_args
                      if k not in _ARG_KEYWORDS and isinstance(getattr(self, k), ObjState)}

            # initialize code namespace
            used_args, code_arg2call, code_lines = set(), {}, []
            func_code, code_scope = self.__step_substitute_integrator(func)

            # check function code
            add_args = set()
            for i, arg in enumerate(func_args):
                used_args.add(arg)
                if len(states) == 0:
                    continue
                if arg in states:
                    st = states[arg]
                    var2idx = st['_var2idx']
                    for st_k in st._keys:
                        p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                        r = f"{arg}[{var2idx[st_k]}]"
                        func_code = re.sub(r'' + p, r, func_code)
                    p = f'{arg}.pull_cond()'
                    if p in func_code:
                        r = f'{arg}[{self.name}_{arg}_dout]'
                        func_code = func_code.replace(p, r)
                        add_args.add(f'{self.name}_{arg}_dout')
                        code_arg2call[f'{self.name}_{arg}_dout'] = f'{self.name}.{arg}._delay_out'
                    p = f'{arg}.push_cond'
                    if p in func_code:
                        res = re.findall(r'(' + p + r'\((\w+?)\))', func_code)
                        if len(res) > 1:
                            raise ModelDefError(f'Cannot call "{p}()" {len(res)} times. Error in code:\n\n'
                                                f'{func_code}')
                        if len(res[0]) != 2:
                            raise ValueError(f'Python regex error for search of "{p}" in code:\n\n{func_code}')
                        res = res[0]
                        text = f'{arg}[{self.name}_{arg}_din] = {res[1]}'
                        func_code = func_code.replace(res[0], text)
                        add_args.add(f'{self.name}_{arg}_din')
                        code_arg2call[f'{self.name}_{arg}_din'] = f'{self.name}.{arg}._delay_in'

            # substitute arguments
            code_args = add_args
            # code_args = set()
            arg_substitute = {}
            for arg in used_args:
                if arg in _ARG_KEYWORDS:
                    new_arg = arg
                    code_arg2call[arg] = arg
                else:
                    new_arg = f'{self.name}_{arg}'
                    arg_substitute[arg] = new_arg
                    if isinstance(getattr(self, arg), ObjState):
                        code_arg2call[new_arg] = f'{self.name}.{arg}["_data"]'
                    else:
                        code_arg2call[new_arg] = f'{self.name}.{arg}'
                code_args.add(new_arg)
            func_code = tools.word_replace(func_code, arg_substitute)

            # final
            code_lines = func_code.split('\n')

            code_lines.insert(0, f'# "{func_name}" step function of {self.name}')
            self._codegen[func_name] = {'scopes': code_scope, 'args': code_args,
                                        'arg2calls': code_arg2call, 'codes': code_lines}

    def __step_mode_nb_single_neu(self):
        func = self._steps[0]

        # get code scope
        used_args, code_arg2call, code_lines = set(), {}, []
        func_code, code_scope = self.__step_substitute_integrator(func)
        func_args = inspect.getfullargspec(func).args
        states = {k: getattr(self, k) for k in func_args
                  if k not in _ARG_KEYWORDS and isinstance(getattr(self, k), NeuState)}

        # update parameters in code scope
        for p, v in self.pars_update.items():
            if p in code_scope:
                code_scope[p] = v
        for p_k in self._hetero_pars.keys():
            if p_k not in code_scope:
                raise ValueError(f'Heterogeneous parameter "{p_k}" is not in '
                                 f'main function, it will not work.')

        # update functions in code scope
        for k, v in code_scope.items():
            if callable(v):
                code_scope[k] = tools.numba_func(func=v, params=self.pars_update)

        # check function code
        for i, arg in enumerate(func_args):
            used_args.add(arg)
            if len(states) == 0:
                continue
            if arg in states.keys():
                st = states[arg]
                var2idx = st['_var2idx']
                for st_k in st._keys:
                    p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                    r = f"{arg}[{var2idx[st_k]}, _ni_]"
                    func_code = re.sub(r'' + p, r, func_code)

        # substitute arguments
        code_args = set()
        arg_substitute = {}
        for arg in used_args:
            if arg in _ARG_KEYWORDS:
                new_arg = arg
                code_arg2call[arg] = arg
            else:
                new_arg = f'{self.name}_{arg}'
                arg_substitute[arg] = new_arg
                if isinstance(getattr(self, arg), NeuState):
                    code_arg2call[new_arg] = f'{self.name}.{arg}["_data"]'
                else:
                    code_arg2call[new_arg] = f'{self.name}.{arg}'
            code_args.add(new_arg)

        # substitute multi-dimensional parameter "p" to "p[_ni_]"
        for p in self._hetero_pars.keys():
            if p in code_scope:
                arg_substitute[p] = f'{p}[_ni_]'

        # substitute
        func_code = tools.word_replace(func_code, arg_substitute)

        # final
        code_lines = [f'# "update" step function of {self.name}',
                      f'for _ni_ in numba.prange({self.num}):']
        code_lines.extend(['  ' + l for l in func_code.split('\n')])
        code_scope['numba'] = import_module('numba')
        self._codegen['update'] = {'scopes': code_scope, 'args': code_args,
                                   'arg2calls': code_arg2call, 'codes': code_lines}

    def __step_mode_nb_single_syn(self):
        # get the delay variable
        output_code = tools.get_main_code(self._steps[1])
        delay_keys = list(set(re.findall(r'ST\[[\'"](\w+)[\'"]\]', output_code)))
        self.set_ST(_SynStateForNbSingleMode(self.ST._vars)(self.num, self.delay_len, delay_keys))

        for i, func in enumerate(self._steps):
            func_name = 'update' if i == 0 else 'output'

            # get code scope
            used_args, code_arg2call, code_lines = set(), {}, []
            func_args = inspect.getfullargspec(func).args
            func_code, code_scope = self.__step_substitute_integrator(func)
            states = {k: getattr(self, k) for k in func_args
                      if k not in _ARG_KEYWORDS and isinstance(getattr(self, k), ObjState)}

            # update parameters in code scope
            for p, v in self.pars_update.items():
                if p in code_scope:
                    code_scope[p] = v
            for p_k in self._hetero_pars.keys():
                if p_k not in code_scope:
                    raise ValueError(f'Heterogeneous parameter "{p_k}" is not in '
                                     f'main function, it will not work.')

            # update functions in code scope
            for k, v in code_scope.items():
                if callable(v):
                    code_scope[k] = tools.numba_func(func=v, params=self.pars_update)

            # check function code
            add_args = set()
            if func_name == 'update':
                add_args.add(f'{self.name}_din')
                code_arg2call[f'{self.name}_din'] = f'{self.name}.ST._delay_in'
                add_args.add(f'{self.name}_din2')
                code_arg2call[f'{self.name}_din2'] = f'{self.name}.ST._delay_in2'
            else:
                add_args.add(f'{self.name}_dout')
                code_arg2call[f'{self.name}_dout'] = f'{self.name}.ST._delay_out'
            for i, arg in enumerate(func_args):
                used_args.add(arg)
                if len(states) == 0:
                    continue
                if arg in states:
                    st = states[arg]
                    var2idx = st['_var2idx']
                    if arg == 'ST':
                        for st_k in st._keys:
                            if st_k in st._delay_offset:
                                if func_name == 'update':
                                    idx = f'{self.name}_din2'
                                    p = f'(=.*)ST\[([\'"]{st_k}[\'"])\]'
                                    r = f"\\1{arg}[{var2idx[st_k]} + {idx}, _syn_i_]"
                                    func_code = re.sub(r'' + p, r'' + r, func_code)
                                    idx = f'{self.name}_din'
                                    p = f'\\bST\[([\'"]{st_k}[\'"])\](\s+=)'
                                    r = f"{arg}[{var2idx[st_k]} + {idx}, _syn_i_]\\2"
                                    func_code = re.sub(r'' + p, r'' + r, func_code)
                                else:
                                    idx = f'{self.name}_dout'
                                    p = f'ST\[([\'"]{st_k}[\'"])\]'
                                    r = f"{arg}[{var2idx[st_k]} + {idx}, _syn_i_]"
                                    func_code = re.sub(r'' + p, r'' + r, func_code)
                            else:
                                p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                                r = f"{arg}[{var2idx[st_k]}, _syn_i_]"
                                func_code = re.sub(r'' + p, r, func_code)
                    elif arg == 'pre':
                        for st_k in st._keys:
                            p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                            r = f"{arg}[{var2idx[st_k]}, _pre_i_]"
                            func_code = re.sub(r'' + p, r, func_code)
                    elif arg == 'post':
                        for st_k in st._keys:
                            p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                            r = f"{arg}[{var2idx[st_k]}, _post_i_]"
                            func_code = re.sub(r'' + p, r, func_code)
                    else:
                        raise ValueError

            # substitute arguments
            code_args = add_args
            arg_substitute = {}
            for arg in used_args:
                if arg in _ARG_KEYWORDS:
                    new_arg = arg
                    code_arg2call[arg] = arg
                else:
                    new_arg = f'{self.name}_{arg}'
                    arg_substitute[arg] = new_arg
                    if isinstance(getattr(self, arg), ObjState):
                        code_arg2call[new_arg] = f'{self.name}.{arg}["_data"]'
                    else:
                        code_arg2call[new_arg] = f'{self.name}.{arg}'
                code_args.add(new_arg)
            # substitute multi-dimensional parameter "p" to "p[_ni_]"
            for p in self._hetero_pars.keys():
                if p in code_scope:
                    arg_substitute[p] = f'{p}[_ni_]'
            # substitute
            func_code = tools.word_replace(func_code, arg_substitute)

            # final
            assert 'ST' in states
            has_pre = 'pre' in states
            has_post = 'post' in states
            if has_pre and has_post:
                code_args.add(f'{self.name}_post_idx')
                code_arg2call[f'{self.name}_post_idx'] = f'{self.name}.post_idx'
                code_args.add(f'{self.name}_pre2syn')
                code_arg2call[f'{self.name}_pre2syn'] = f'{self.name}.pre2syn'
                code_lines = [f'# "{func_name}" step function of {self.name}',
                              f'for _pre_i_ in numba.prange({self.pre_group.num}):',
                              f'  for _syn_i_ in {self.name}_pre2syn[_pre_i_]:',
                              f'    _post_i_ = {self.name}_post_idx[_syn_i_]']
                blank = '  ' * 3
            elif has_pre:
                code_args.add(f'{self.name}_pre2syn')
                code_arg2call[f'{self.name}_pre2syn'] = f'{self.name}.pre2syn'
                code_lines = [f'# "{func_name}" step function of {self.name}',
                              f'for _pre_i_ in numba.prange({self.pre_group.num}):',
                              f'  for _syn_i_ in {self.name}_pre2syn[_pre_i_]:']
                blank = '  ' * 2
            elif has_post:
                code_args.add(f'{self.name}_post2syn')
                code_arg2call[f'{self.name}_post2syn'] = f'{self.name}.post2syn'
                code_lines = [f'# "{func_name}" step function of {self.name}',
                              f'for _post_i_ in numba.prange({self.post_group.num}):',
                              f'  for _syn_i_ in {self.name}_post2syn[_post_i_]:']
                blank = '  ' * 2
            else:
                code_lines = [f'# "{func_name}" step function of {self.name}',
                              f'for _syn_i_ in numba.prange({self.num}):']
                blank = '  ' * 1

            code_lines.extend([blank + l for l in func_code.split('\n')])
            code_scope['numba'] = import_module('numba')
            self._codegen[func_name] = {'scopes': code_scope, 'args': code_args,
                                        'arg2calls': code_arg2call, 'codes': code_lines}

    def _add_steps(self):
        if profile.is_numpy_bk():
            if self.model.vector_based:
                self.__step_mode_np_group()
            else:
                self.__step_mode_np_single()

        elif profile.is_numba_bk():
            if self.model.vector_based:
                self.__step_mode_nb_group()
            else:
                if self._cls_type == _NEU_GROUP:
                    self.__step_mode_nb_single_neu()
                else:
                    self.__step_mode_nb_single_syn()

        else:
            raise NotImplementedError

    def _add_input(self, key_val_ops_types):
        code_scope, code_args, code_arg2call, code_lines = {self.name: self}, set(), {}, []
        input_idx = 0

        # check datatype of the input
        # ----------------------------
        has_iter = False
        for _, _, _, t in key_val_ops_types:
            try:
                assert t in ['iter', 'fix']
            except AssertionError:
                raise ModelUseError('Only support inputs of "iter" and "fix" types.')
            if t == 'iter':
                has_iter = True
        if has_iter:
            code_args.add('_i_')
            code_arg2call['_i_'] = '_i_'

        # check data operations
        # ----------------------
        for _, _, ops, _ in key_val_ops_types:
            try:
                assert ops in ['-', '+', 'x', '/', '=']
            except AssertionError:
                raise ModelUseError('Only support five input operations: +, -, x, /, =')
        ops2str = {'-': 'sub', '+': 'add', 'x': 'mul', '/': 'div', '=': 'assign'}

        # generate code of input function
        # --------------------------------
        for key, val, ops, data_type in key_val_ops_types:
            attr_item = key.split('.')

            # get the left side #
            if len(attr_item) == 1 and (attr_item[0] not in self.ST):  # if "item" is the model attribute
                attr, item = attr_item[0], ''
                try:
                    assert hasattr(self, attr)
                except AssertionError:
                    raise ModelUseError(f'Model "{self.name}" doesn\'t have "{attr}" attribute", '
                                        f'and "{self.name}.ST" doesn\'t have "{attr}" field.')
                try:
                    assert isinstance(getattr(self, attr), np.ndarray)
                except AssertionError:
                    raise ModelUseError(f'NumpyBrain only support input to arrays.')

                if profile.is_numpy_bk():
                    left = f'{self.name}.{attr}'
                else:
                    left = f'{self.name}_{attr}'
                    code_args.add(left)
                    code_arg2call[left] = f'{self.name}.{attr}'
            else:
                if len(attr_item) == 1:
                    attr, item = 'ST', attr_item[0]
                elif len(attr_item) == 2:
                    attr, item = attr_item[0], attr_item[1]
                else:
                    raise ModelUseError(f'Unknown target : {key}.')
                try:
                    assert item in getattr(self, attr)
                except AssertionError:
                    raise ModelUseError(f'"{self.name}.{attr}" doesn\'t have "{item}" field.')

                if profile.is_numpy_bk():
                    left = f'{self.name}.{attr}["{item}"]'
                else:
                    idx = getattr(self, attr)['_var2idx'][item]
                    left = f'{self.name}_{attr}[{idx}]'
                    code_args.add(f'{self.name}_{attr}')
                    code_arg2call[f'{self.name}_{attr}'] = f'{self.name}.{attr}["_data"]'

            # get the right side #
            right = f'{self.name}_input{input_idx}_{attr}_{item}_{ops2str[ops]}'
            code_scope[right] = val
            if data_type == 'iter':
                right = right + '[_i_]'
            input_idx += 1

            # final code line #
            if ops == '=':
                code_lines.append(left + " = " + right)
            else:
                code_lines.append(left + f" {ops}= " + right)

        # final code
        # ----------
        if len(key_val_ops_types) > 0:
            code_lines.insert(0, f'# "input" step function of {self.name}')

        if profile.is_numpy_bk():
            if len(key_val_ops_types) > 0:
                code_args = sorted(list(code_args))
                code_lines.insert(0, f'\ndef input_step({", ".join(code_args)}):')

                # compile function
                func_code = '\n  '.join(code_lines)
                if profile.auto_pep8:
                    func_code = autopep8.fix_code(func_code)
                exec(compile(func_code, '', 'exec'), code_scope)
                self.input_step = code_scope['input_step']

                # format function call
                code_arg2call = [code_arg2call[arg] for arg in code_args]
                func_call = f'{self.name}.input_step({", ".join(code_arg2call)})'

                if profile.show_codgen:
                    print(func_code)
                    print()
                    code_scope.pop('__builtins__')
                    code_scope.pop('input_step')
                    pprint(code_scope)
                    print()
            else:
                self.input_step = None
                func_call = ''

            self._codegen['input'] = {'func': self.input_step, 'call': func_call}

        else:
            self._codegen['input'] = {'scopes': code_scope, 'args': code_args,
                                      'arg2calls': code_arg2call, 'codes': code_lines}

    def _add_monitor(self, run_length):
        code_scope, code_args, code_arg2call, code_lines = {self.name: self}, set(), {}, []
        idx_no = 0

        # generate code of monitor function
        # ---------------------------------
        for key, indices in self._mon_vars:
            # check indices #
            if indices is not None:
                if isinstance(indices, list):
                    try:
                        isinstance(indices[0], int)
                    except AssertionError:
                        raise ModelUseError('Monitor index only supports list [int] or 1D array.')
                elif isinstance(indices, np.ndarray):
                    try:
                        assert np.ndim(indices) == 1
                    except AssertionError:
                        raise ModelUseError('Monitor index only supports list [int] or 1D array.')
                else:
                    raise ModelUseError(f'Unknown monitor index type: {type(indices)}.')

            attr_item = key.split('.')

            # get the code line #
            if (len(attr_item) == 1) and (attr_item[0] not in getattr(self, 'ST')):
                attr = attr_item[0]
                try:
                    assert hasattr(self, attr)
                except AssertionError:
                    raise ModelUseError(f'Model "{self.name}" doesn\'t have "{attr}" attribute", '
                                        f'and "{self.name}.ST" doesn\'t have "{attr}" field.')
                try:
                    assert isinstance(getattr(self, attr), np.ndarray)
                except AssertionError:
                    assert ModelUseError(f'NumpyBrain only support monitor of arrays.')

                shape = getattr(self, attr).shape

                idx_name = f'{self.name}_idx{idx_no}_{attr}'
                if profile.is_numpy_bk():
                    if indices is None:
                        line = f'{self.name}.mon["{key}"][i] = {self.name}.{attr}'
                    else:
                        line = f'{self.name}.mon["{key}"][i] = {self.name}.{attr}[{idx_name}]'
                        code_scope[idx_name] = indices
                        idx_no += 1

                else:
                    mon_name = f'{self.name}_mon_{attr}'
                    target_name = f'{self.name}_{attr}'
                    if indices is None:
                        line = f'{mon_name}[_i_] = {target_name}'
                    else:
                        line = f'{mon_name}[_i_] = {target_name}[{idx_name}]'
                        code_scope[idx_name] = indices
                        idx_no += 1
                    code_args.add(mon_name)
                    code_arg2call[mon_name] = f'{self.name}.mon["{key}"]'
                    code_args.add(target_name)
                    code_arg2call[target_name] = f'{self.name}.{attr}'
            else:
                if len(attr_item) == 1:
                    item, attr = attr_item[0], 'ST'
                elif len(attr_item) == 2:
                    attr, item = attr_item
                else:
                    raise ModelUseError(f'Unknown target : {key}.')

                if not self.model.vector_based and self._cls_type == _SYN_CONN and \
                        attr == 'ST' and profile.is_numba_bk():
                    shape = getattr(self, attr)[item][0].shape
                else:
                    shape = getattr(self, attr)[item].shape

                idx_name = f'{self.name}_idx{idx_no}_{attr}_{item}'
                if profile.is_numpy_bk():
                    if indices is None:
                        line = f'{self.name}.mon["{key}"][_i_] = {self.name}.{attr}["{item}"]'
                    else:
                        line = f'{self.name}.mon["{key}"][_i_] = {self.name}.{attr}["{item}"][{idx_name}]'
                        code_scope[idx_name] = indices
                        idx_no += 1
                else:
                    idx = getattr(self, attr)['_var2idx'][item]
                    mon_name = f'{self.name}_mon_{attr}_{item}'
                    target_name = f'{self.name}_{attr}'
                    if not self.model.vector_based and self._cls_type == _SYN_CONN and \
                            attr == 'ST' and item in self.ST._delay_offset:
                        code_args.add(f'{self.name}_din')
                        code_arg2call[f'{self.name}_din'] = f'{self.name}.ST._delay_in'
                        if indices is None:
                            line = f'{mon_name}[_i_] = {target_name}[{idx} + {self.name}_din]'
                        else:
                            line = f'{mon_name}[_i_] = {target_name}[{idx} + {self.name}_din][{idx_name}]'
                            code_scope[idx_name] = indices
                    else:
                        if indices is None:
                            line = f'{mon_name}[_i_] = {target_name}[{idx}]'
                        else:
                            line = f'{mon_name}[_i_] = {target_name}[{idx}][{idx_name}]'
                            code_scope[idx_name] = indices
                    idx_no += 1
                    code_args.add(mon_name)
                    code_arg2call[mon_name] = f'{self.name}.mon["{key}"]'
                    code_args.add(target_name)
                    code_arg2call[target_name] = f'{self.name}.{attr}["_data"]'

            # initialize monitor array #
            if indices is None:
                self.mon[key] = np.zeros((run_length,) + shape, dtype=np.float_)
            else:
                self.mon[key] = np.zeros((run_length, len(indices)) + shape[1:], dtype=np.float_)

            # add line #
            code_lines.append(line)

        # final code
        # ----------
        if len(self._mon_vars):
            code_args.add('_i_')
            code_arg2call['_i_'] = '_i_'
            code_lines.insert(0, f'# "monitor" step function of {self.name}')

        if profile.is_numpy_bk():
            if len(self._mon_vars):
                code_args = sorted(list(code_args))
                code_lines.insert(0, f'\ndef monitor_step({", ".join(code_args)}):')

                # compile function
                func_code = '\n  '.join(code_lines)
                if profile.auto_pep8:
                    func_code = autopep8.fix_code(func_code)
                exec(compile(func_code, '', 'exec'), code_scope)
                self.monitor_step = code_scope['monitor_step']

                # format function call
                code_arg2call = [code_arg2call[arg] for arg in code_args]
                func_call = f'{self.name}.monitor_step({", ".join(code_arg2call)})'

                if profile.show_codgen:
                    print(func_code)
                    print()
                    code_scope.pop('__builtins__')
                    code_scope.pop('monitor_step')
                    pprint(code_scope)
                    print()
            else:
                self.monitor_step = None
                func_call = ''

            self._codegen['monitor'] = {'func': self.monitor_step, 'call': func_call}

        else:
            self._codegen['monitor'] = {'scopes': code_scope, 'args': code_args,
                                        'arg2calls': code_arg2call, 'codes': code_lines}

    def _merge_steps(self):
        codes_of_calls = []  # call the compiled functions
        if profile.is_numpy_bk():  # numpy mode
            for item in self._schedule:
                if item in self._codegen:
                    func_call = self._codegen[item]['call']
                    if func_call:
                        codes_of_calls.append(func_call)

        else:  # non-numpy mode
            lines, code_scopes, args, arg2calls = [], dict(), set(), dict()
            for item in self._schedule:
                if item in self._codegen:
                    lines.extend(self._codegen[item]['codes'])
                    code_scopes.update(self._codegen[item]['scopes'])
                    args = args | self._codegen[item]['args']
                    arg2calls.update(self._codegen[item]['arg2calls'])

            args = sorted(list(args))
            arg2calls_list = [arg2calls[arg] for arg in args]
            lines.insert(0, f'\n# {self.name} "merge_func"'
                            f'\ndef merge_func({", ".join(args)}):')
            func_code = '\n  '.join(lines)
            if profile.auto_pep8:
                func_code = autopep8.fix_code(func_code)
            exec(compile(func_code, '', 'exec'), code_scopes)

            self.merge_func = code_scopes['merge_func']
            func_call = f'{self.name}.merge_func({", ".join(arg2calls_list)})'
            codes_of_calls.append(func_call)

            if profile.show_codgen:
                print(func_code)
                print()
                code_scopes.pop('__builtins__')
                code_scopes.pop('merge_func')
                pprint(code_scopes)
                print()

        return codes_of_calls

    def set_ST(self, new_ST):
        type_checker = self.model.requires['ST']
        try:
            type_checker.check(new_ST)
        except TypeMismatchError:
            raise ModelUseError(f'"new_ST" doesn\'t satisfy TypeChecker "{str(type_checker)}".')
        super(BaseEnsemble, self).__setattr__('ST', new_ST)

    @property
    def requires(self):
        return self.model.requires

    @property
    def _keywords(self):
        kws = [
            # attributes
            'model', 'num', 'ST', 'PA', 'vars_init', 'pars_update', '_mon_vars',
            'mon', '_cls_type', '_codegen', '_keywords', 'steps', '_schedule',
            # self functions
            '_merge_steps', '_add_steps', '_add_input', '_add_monitor',
            'get_schedule', 'set_schedule'
        ]
        if hasattr(self, 'model'):
            kws += self.model.step_names
        return kws

    def get_schedule(self):
        return self._schedule

    def set_schedule(self, schedule):
        try:
            assert isinstance(schedule, (list, tuple))
        except AssertionError:
            raise ModelUseError('"schedule" must be a list/tuple.')
        all_func_names = ['input', 'monitor'] + self.model.step_names
        for s in schedule:
            try:
                assert s in all_func_names
            except AssertionError:
                raise ModelUseError(f'Unknown step function "{s}" for "{self._cls_type}" model.')
        super(BaseEnsemble, self).__setattr__('_schedule', schedule)

    def __setattr__(self, key, value):
        if key in self._keywords:
            if hasattr(self, key):
                raise KeyError(f'"{key}" is a keyword in "{self._cls_type}" model, please change another name.')
        super(BaseEnsemble, self).__setattr__(key, value)
