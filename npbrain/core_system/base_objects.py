# -*- coding: utf-8 -*-

import inspect
import re
from importlib import import_module

import autopep8

from .types import NeuState
from .types import ObjState
from .types import SynState
from .types import TypeChecker
from .types import TypeMismatchError
from .. import numpy as np
from .. import profile
from .. import tools
from ..integration import Integrator
from ..integration.sympy_tools import get_mapping_scope

__all__ = [
    # errors
    'ModelDefError',
    'ModelUseError',

    # base types
    'BaseType',
    'BaseEnsemble',
]

_ARG_KEYWORDS = ['_dt_', '_t_', '_i_']
_NEU_GROUP = 'NeuGroup'
_SYN_CONN = 'SynConn'


class ModelDefError(Exception):
    """Model definition error."""
    pass


class ModelUseError(Exception):
    """Model use error."""
    pass


class BaseType(object):
    """The base type of neuron and synapse.

    Parameters
    ----------
    name : str, optional
        Model name.
    vector_based : bool
        Whether the model is written in the neuron-group level or in the single-neuron level.
    """

    def __init__(self, requires, steps, name, vector_based=True, heter_params_replace=None):
        # type : neuron based or group based code
        # ---------------------------------------
        self.vector_based = vector_based

        # name
        # -----
        self.name = name

        # requires
        # -----------
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

        # variables
        # ----------
        self.variables = self.requires['ST']._vars

        # steps
        # ------
        self.steps, self.step_names, self.steps_scope = [], [], dict()
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
            func_name = tools.get_func_name(func, replace=True)
            self.step_names.append(func_name)
            self.steps.append(func)
            scope = tools.get_func_scope(func, include_dispatcher=True)
            self.steps_scope.update(scope)
            setattr(self, func_name, func)

        # heterogeneous parameter replace
        # --------------------------------
        if heter_params_replace is None:
            heter_params_replace = dict()
        try:
            assert isinstance(heter_params_replace, dict)
        except AssertionError:
            raise ModelDefError('"heter_params_replace" must be a dict.')
        self.heter_params_replace = heter_params_replace

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
    name : str
        Name of the (neurons/synapses) ensemble.
    num : int
        The number of the neurons/synapses.
    model : BaseType
        The (neuron/synapse) model.
    monitors : list, tuple, None
        Variables to monitor.
    pars_update : dict, None
        Parameters to update.
    cls_type : str
        Class type.
    """

    def __init__(self, name, num, model, monitors, pars_update, cls_type):
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
            raise ModelUseError(f'"{self.name}" isn\'t a valid identifier according to Python '
                                f'language definition. Please choose another name.')

        # num
        # ---
        self.num = num

        # parameters
        # ----------
        self._hetero_pars = {}
        pars_update = dict() if pars_update is None else pars_update
        try:
            assert isinstance(pars_update, dict)
        except AssertionError:
            raise ModelUseError('"pars_update" must be a dict.')
        for k, v in pars_update.items():
            val_size = np.size(v)
            if val_size != 1:
                if val_size != num:
                    raise ModelUseError(f'The size of parameter "{k}" is wrong, "{val_size}" != 1 '
                                        f'and "{val_size}" != {num}.')
                else:
                    self._hetero_pars[k] = v
            if k not in model.steps_scope:
                raise ModelUseError(f'Parameter "{k}" is not defined in "{model.name}" variable scope.\n'
                                    f'This may be caused by that "{k}" is used to compute an intermediate variable, \n'
                                    f'but not directly used by the function. Therefore, NumpyBrain cannot update \n'
                                    f'this parameter.')
        self._pars_to_update = pars_update

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
        for i, func in enumerate(self.model.steps):
            for arg in inspect.getfullargspec(func).args:
                if not (arg in _ARG_KEYWORDS + ['self']) and not hasattr(self, arg):
                    raise ModelUseError(f'Function "{tools.get_func_name(func, replace=True)}" in "{self.model.name}" '
                                        f'requires "{arg}" as argument, but "{arg}" is not defined in "{self.name}".')

    def __step_delay_keys(self):
        delay_keys = set()
        if self._cls_type == _SYN_CONN:
            # check "delayed" decorator
            delay_funcs = []
            for func in self.model.steps:
                if func.__name__.startswith('_npbrain_delayed_'):
                    delay_funcs.append(func)

            # get delayed variables
            if len(delay_funcs):
                delay_func_code = '\n'.join([tools.get_main_code(func) for func in delay_funcs])
                delay_func_code_left = '\n'.join([line.split('=')[0] for line in tools.get_code_lines(delay_func_code)])
                delay_keys_in_left = set(re.findall(r'ST\[[\'"](\w+)[\'"]\]', delay_func_code_left))
                if len(delay_keys_in_left) > 0:
                    raise ModelDefError('Delayed function cannot assign value to "ST".')
                delay_keys = set(re.findall(r'ST\[[\'"](\w+)[\'"]\]', delay_func_code))
                self.set_ST(self.ST.make_copy(self.num, self.delay_len, list(delay_keys)))
        return delay_keys

    def __step_mode_np_group(self):
        # check whether the model include heterogeneous parameters
        if len(self._hetero_pars) > 0:
            raise ModelUseError(f'This model has heterogeneous parameters '
                                f'"{list(self._hetero_pars.keys())}", '
                                f'it cannot be compiled in numpy mode.')
        # get the delay keys
        delay_keys = self.__step_delay_keys()

        for func in self.model.steps:
            func_name = func.__name__
            func_name_stripped = tools.get_func_name(func, replace=True)
            func_args = inspect.getfullargspec(func).args

            if 'ST' in func_args and len(delay_keys) > 0:

                if func_name.startswith('_npbrain_delayed_'):
                    # In the delayed function,
                    # synapse state should pull out from the delay queues
                    func_code = tools.get_main_code(func)
                    func_delay_keys = set(re.findall(r'ST\[[\'"](\w+)[\'"]\]', func_code))
                    code_scope = {func_name_stripped: func}
                    code_lines = [f'def {func_name_stripped}_enhanced({", ".join(func_args)}):',
                                  f'  new_ST = dict()']
                    for key in func_delay_keys:
                        code_lines.append(f'  new_ST["{key}"] = ST.delay_pull("{key}")')
                    code_lines.append('  ST = new_ST')
                    code_lines.append(f'  {func_name_stripped}({", ".join(func_args)})')

                else:
                    # In other un-delayed function,
                    # the calculated values of delayed keys should be push into the delay queues
                    func_code = tools.get_main_code(func)
                    func_keys = set(re.findall(r'ST\[[\'"](\w+)[\'"]\]', func_code))
                    func_delay_keys = func_keys.intersection(delay_keys)
                    if len(func_delay_keys) > 0:
                        code_scope = {func_name_stripped: func}
                        code_lines = [f'def {func_name_stripped}_enhanced({", ".join(func_args)}):',
                                      f'  {func_name_stripped}({", ".join(func_args)})']
                        for key in func_delay_keys:
                            code_lines.append(f'  ST.delay_push(ST["{key}"], var="{key}")')
                    else:
                        code_lines = []
                        code_scope = {}

                if len(code_lines):
                    # Compile the modified step function
                    func_code = '\n'.join(code_lines)
                    if profile._auto_pep8:
                        func_code = autopep8.fix_code(func_code)
                    exec(compile(func_code, '', 'exec'), code_scope)
                    func = code_scope[func_name_stripped + '_enhanced']

                    if profile._show_formatted_code:
                        tools.show_code_str(func_code)
                        tools.show_code_scope(code_scope, ['__builtins__', func_name_stripped])

            # set the function to the this model
            setattr(self, func_name_stripped, func)

            # get the function call
            arg_calls = []
            for arg in func_args:
                if arg in _ARG_KEYWORDS:
                    arg_calls.append(arg)
                else:
                    arg_calls.append(f"{self.name}.{arg}")
            func_call = f'{self.name}.{func_name_stripped}({", ".join(arg_calls)})'

            # get the function result
            self._codegen[func_name_stripped] = {'func': func, 'call': func_call}

    def __step_mode_np_single(self):
        # check number of the neurons/synapses,
        # too huge number of neurons/synapses sharply reduce running speed
        if self.num > 4000:
            raise ModelUseError(f'The number of the '
                                f'{"neurons" if self._cls_type == _NEU_GROUP else "synapses"} is '
                                f'too huge (>4000), please use numba backend or define vector_based model.')

        # check whether the model include heterogeneous parameters
        if len(self._hetero_pars) > 0:
            raise ModelUseError(f'This model has heterogeneous parameters '
                                f'"{list(self._hetero_pars.keys())}", '
                                f'it cannot be compiled in numpy mode.')

        # get the delay keys
        delay_keys = self.__step_delay_keys()

        for func in self.model.steps:
            func_name = func.__name__
            func_name_stripped = tools.get_func_name(func, replace=True)

            # function argument
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
            code_scope = {f'{func_name_stripped}_origin': func}

            # codes
            has_ST = 'ST' in state_args
            has_pre = 'pre' in state_args
            has_post = 'post' in state_args
            if has_ST:  # have ST
                if has_pre and has_post:
                    code_arg.extend(['pre2syn', 'post_ids'])
                    code_arg2call['pre2syn'] = f'{self.name}.pre2syn'
                    code_arg2call['post_ids'] = f'{self.name}.post_ids'

                    code_lines = [f'def {func_name_stripped}({", ".join(code_arg)}):',
                                  f'  for pre_idx in range({self.pre_group.num}):',
                                  f'    pre = {self.name}_pre.extract_by_index(pre_idx)',
                                  f'    for _obj_i_ in pre2syn[pre_idx]:',
                                  f'      post_i = post_ids[_obj_i_]',
                                  f'      post = {self.name}_post.extract_by_index(post_i)']
                    prefix = '  ' * 3
                elif has_pre:
                    code_arg.append('pre2syn')
                    code_arg2call['pre2syn'] = f'{self.name}.pre2syn'

                    code_lines = [f'def {func_name_stripped}({", ".join(code_arg)}):',
                                  f'  for pre_idx in range({self.pre_group.num}):',
                                  f'    pre = {self.name}_pre.extract_by_index(pre_idx)',
                                  f'    for _obj_i_ in pre2syn[pre_idx]:']
                    prefix = '  ' * 3
                elif has_post:
                    code_arg.append('post2syn')
                    code_arg2call['post2syn'] = f'{self.name}.post2syn'

                    code_lines = [f'def {func_name_stripped}({", ".join(code_arg)}):',
                                  f'  for post_id in range({self.post_group.num}):',
                                  f'    post = {self.name}_post.extract_by_index(post_id)',
                                  f'    for _obj_i_ in post2syn[post_id]:']
                    prefix = '  ' * 3
                else:
                    code_lines = [f'def {func_name_stripped}({", ".join(code_arg)}):',
                                  f'  for _obj_i_ in range({self.num}):']
                    prefix = '  ' * 2

                if func_name.startswith('_npbrain_delayed_'):
                    # Function with "delayed" decorator should use ST pulled from the delay queue
                    code_lines.append(prefix + f'ST = {self.name}_ST.extract_by_index(_obj_i_, delay_pull=True)')
                    code_lines.append(prefix + f'{func_name_stripped}_origin({", ".join(func_args)})')
                else:
                    # Other function with "delayed" decorator
                    code_lines.append(prefix + f'ST = {self.name}_ST.extract_by_index(_obj_i_)')
                    code_lines.append(prefix + f'{func_name_stripped}_origin({", ".join(func_args)})')
                    code_lines.append(prefix + f'{self.name}_ST.update_by_index(_obj_i_, ST)')
                    if len(delay_keys):
                        # Function without "delayed" decorator should push their
                        # updated ST to the delay queue
                        func_code = tools.get_main_code(func)
                        func_keys = set(re.findall(r'ST\[[\'"](\w+)[\'"]\]', func_code))
                        func_delay_keys = func_keys.intersection(delay_keys)
                        if len(func_delay_keys) > 0:
                            for key in func_delay_keys:
                                code_lines.append(f'  {self.name}_ST.delay_push({self.name}_ST["{key}"], "{key}")')

            else:  # doesn't have ST
                try:
                    assert not has_post and not has_pre
                except AssertionError:
                    raise ModelDefError(f'Unknown "{func_name_stripped}" function structure.')
                code_lines = [f'def {func_name_stripped}({", ".join(code_arg)}):',
                              f'  for _obj_i_ in range({self.num}):',
                              f'    {func_name_stripped}_origin({", ".join(func_args)})']

            # append the final results
            code_lines.insert(0, f'# "{func_name_stripped}" step function in {self.name}')

            # compile the updated function
            func_code = '\n'.join(code_lines)
            if profile._auto_pep8:
                func_code = autopep8.fix_code(func_code)
            exec(compile(func_code, '', 'exec'), code_scope)
            func = code_scope[func_name_stripped]
            if profile._show_formatted_code:
                tools.show_code_str(func_code)
                tools.show_code_scope(code_scope, ['__builtins__', func_name_stripped])

            # set the function to the model
            setattr(self, func_name_stripped, func)

            # function call
            func_call = f'{self.name}.{func_name_stripped}({", ".join([code_arg2call[arg] for arg in code_arg])})'

            # final
            self._codegen[func_name_stripped] = {'func': func, 'call': func_call}

    def __step_substitute_integrator(self, func):
        # get code and code lines
        func_code = tools.deindent(tools.get_main_code(func))
        code_lines = tools.get_code_lines(func_code)

        # get function scope
        vars = inspect.getclosurevars(func)
        code_scope = dict(vars.nonlocals)
        code_scope.update(vars.globals)
        code_scope.update({self.name: self})
        if len(code_lines) == 0:
            return '', code_scope

        # code scope update
        scope_to_add = {}
        scope_to_del = set()
        need_add_mapping_scope = False
        for k, v in code_scope.items():
            if isinstance(v, Integrator):
                if profile._merge_integral:
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
                    new_line, args, kwargs = tools.replace_func(line, int_func_name)

                    # append code line of argument replacement
                    func_args = v.diff_eq.func_args
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
                            v2 = tools.numba_func(v2, params=self._pars_to_update)
                        scope_to_add[k2] = v2
                    # noise term (g) is a 1D array
                    g_array = f'_g_{v.py_func_name}'
                    if g_array in v.code_scope:
                        self._hetero_pars[g_array] = v.code_scope[g_array]
                    # deterministic term (f) is a 1D array
                    f_array = f'_f_{v.py_func_name}'
                    if f_array in v.code_scope:
                        self._hetero_pars[f_array] = v.code_scope[f_array]

                else:
                    if not self.model.vector_based:
                        for ks, vs in tools.get_func_scope(v.update_func, include_dispatcher=True).items():
                            if ks in self._hetero_pars:
                                raise ModelUseError(f'Heterogeneous parameter "{ks}" is not in step functions, '
                                                    f'it will not work.\n'
                                                    f'Please try to set "npbrain.profile.merge_integral = True" to '
                                                    f'merge parameter "{ks}" into the step functions.')

                    code_scope[k] = tools.numba_func(v.update_func, params=self._pars_to_update)

        # update code scope
        if need_add_mapping_scope:
            code_scope.update(get_mapping_scope())
        code_scope.update(scope_to_add)
        for k in scope_to_del:
            code_scope.pop(k)

        # return code lines and code scope
        return '\n'.join(code_lines), code_scope

    def __step_mode_nb_group(self):
        # check whether the model include heterogeneous parameters
        delay_keys = self.__step_delay_keys()

        #
        all_heter_pars = set(self._hetero_pars.keys())

        for func in self.model.steps:
            func_name = func.__name__
            func_name_stripped = tools.get_func_name(func, replace=True)
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

                    if self._is_state_attr(arg):
                        # Function with "delayed" decorator should use ST pulled from the delay queue
                        if func_name.startswith('_npbrain_delayed_'):
                            add_args.add(f'{self.name}_dout')
                            code_arg2call[f'{self.name}_dout'] = f'{self.name}.{arg}._delay_out'
                            for st_k in delay_keys:
                                p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                                r = f"{arg}[{var2idx['_' + st_k + '_offset']} + {self.name}_dout]"
                                func_code = re.sub(r'' + p, r, func_code)
                        elif len(delay_keys) > 0:
                            # Function without "delayed" decorator should push their
                            # updated ST to the delay queue
                            func_keys = set(re.findall(r'ST\[[\'"](\w+)[\'"]\]', func_code))
                            func_delay_keys = func_keys.intersection(delay_keys)
                            if len(func_delay_keys) > 0:
                                add_args.add(f'{self.name}_din')
                                code_arg2call[f'{self.name}_din'] = f'{self.name}.{arg}._delay_in'
                                for st_k in func_delay_keys:
                                    right = f'{arg}[{var2idx[st_k]}]'
                                    left = f"{arg}[{var2idx['_' + st_k + '_offset']} + {self.name}_din]"
                                    func_code += f'\n{left} = {right}'

                    #
                    for st_k in st._keys:
                        p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                        r = f"{arg}[{var2idx[st_k]}]"
                        func_code = re.sub(r'' + p, r, func_code)

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
            # substitute parameters
            for k in code_scope.keys():
                if k in self.model.heter_params_replace:
                    arg_substitute[k] = self.model.heter_params_replace[k]
            # substitute
            func_code = tools.word_replace(func_code, arg_substitute)

            # update code scope
            for k in list(code_scope.keys()):
                if k in self._pars_to_update:
                    code_scope[k] = self._pars_to_update[k]
                if k in all_heter_pars:
                    all_heter_pars.remove(k)

            # final
            code_lines = func_code.split('\n')
            code_lines.insert(0, f'# "{func_name_stripped}" step function of {self.name}')
            code_lines.append('\n')
            self._codegen[func_name_stripped] = {'scopes': code_scope, 'args': code_args,
                                                 'arg2calls': code_arg2call, 'codes': code_lines}

        # WARNING: heterogeneous parameter may not in the main step functions
        if len(all_heter_pars) > 0:
            raise ModelDefError(f'Heterogeneous parameters "{list(all_heter_pars)}" are not defined '
                                f'in main step function. NumpyBrain cannot recognize. Please check.')

    def __step_mode_nb_single(self):
        # check whether the model include heterogeneous parameters
        delay_keys = self.__step_delay_keys()

        #
        all_heter_pars = set(self._hetero_pars.keys())

        for i, func in enumerate(self.model.steps):
            func_name = func.__name__

            # get code scope
            used_args, code_arg2call, code_lines = set(), {}, []
            func_args = inspect.getfullargspec(func).args
            func_code, code_scope = self.__step_substitute_integrator(func)
            states = {k: getattr(self, k) for k in func_args
                      if k not in _ARG_KEYWORDS and isinstance(getattr(self, k), ObjState)}

            # update functions in code scope
            for k, v in code_scope.items():
                if callable(v):
                    code_scope[k] = tools.numba_func(func=v, params=self._pars_to_update)

            # check function arguments, whether need to add "_dout" and "_din"
            add_args = set()
            if func_name.startswith('_npbrain_delayed_'):
                # Function with "delayed" decorator should use ST pulled from the delay queue
                add_args.add(f'{self.name}_dout')
                code_arg2call[f'{self.name}_dout'] = f'{self.name}.ST._delay_out'
            elif len(delay_keys) > 0:
                # Function without "delayed" decorator should push their
                # updated ST to the delay queue
                func_keys = set(re.findall(r'ST\[[\'"](\w+)[\'"]\]', func_code))
                func_delay_keys = func_keys.intersection(delay_keys)
                if len(func_delay_keys) > 0:
                    add_args.add(f'{self.name}_din')
                    code_arg2call[f'{self.name}_din'] = f'{self.name}.ST._delay_in'

            # substitute STATE iterm access to index
            for i, arg in enumerate(func_args):
                used_args.add(arg)
                if len(states) == 0:
                    continue
                if arg in states:
                    st = states[arg]
                    var2idx = st['_var2idx']
                    if self._is_state_attr(arg):
                        if func_name.startswith('_npbrain_delayed_'):
                            # Function with "delayed" decorator should use ST pulled from the delay queue
                            for st_k in delay_keys:
                                p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                                r = f"{arg}[{var2idx['_' + st_k + '_offset']} + {self.name}_dout, _obj_i_]"
                                func_code = re.sub(r'' + p, r, func_code)
                        else:
                            for st_k in st._keys:
                                p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                                r = f"{arg}[{var2idx[st_k]}, _obj_i_]"
                                func_code = re.sub(r'' + p, r, func_code)
                    elif arg == 'pre':
                        for st_k in st._keys:
                            p = f'pre\[([\'"]{st_k}[\'"])\]'
                            r = f"pre[{var2idx[st_k]}, _pre_i_]"
                            func_code = re.sub(r'' + p, r, func_code)
                    elif arg == 'post':
                        for st_k in st._keys:
                            p = f'post\[([\'"]{st_k}[\'"])\]'
                            r = f"post[{var2idx[st_k]}, _post_i_]"
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
                    arg_substitute[p] = f'{p}[_obj_i_]'
            # substitute
            func_code = tools.word_replace(func_code, arg_substitute)

            # add the for loop in the start of the main code
            try:
                assert 'ST' in states
            except AssertionError:
                raise ModelUseError("In numba mode, scalar-based model only support function has 'ST' argument.")
            has_pre = 'pre' in states
            has_post = 'post' in states
            if has_pre and has_post:
                code_args.add(f'{self.name}_post_ids')
                code_arg2call[f'{self.name}_post_ids'] = f'{self.name}.post_ids'
                code_args.add(f'{self.name}_pre2syn')
                code_arg2call[f'{self.name}_pre2syn'] = f'{self.name}.pre2syn'
                code_lines = [f'for _pre_i_ in numba.prange({self.pre_group.num}):',
                              f'  for _syn_i_ in {self.name}_pre2syn[_pre_i_]:',
                              f'    _obj_i_ = {self.name}_post_idx[_syn_i_]']
                blank = '  ' * 2
            elif has_pre:
                code_args.add(f'{self.name}_pre2syn')
                code_arg2call[f'{self.name}_pre2syn'] = f'{self.name}.pre2syn'
                code_lines = [f'for _pre_i_ in numba.prange({self.pre_group.num}):',
                              f'  for _obj_i_ in {self.name}_pre2syn[_pre_i_]:']
                blank = '  ' * 2
            elif has_post:
                code_args.add(f'{self.name}_post2syn')
                code_arg2call[f'{self.name}_post2syn'] = f'{self.name}.post2syn'
                code_lines = [f'for _post_i_ in numba.prange({self.post_group.num}):',
                              f'  for _obj_i_ in {self.name}_post2syn[_post_i_]:']
                blank = '  ' * 2
            else:
                code_lines = [f'for _obj_i_ in numba.prange({self.num}):']
                blank = '  ' * 1

            # add the main code (user defined)
            code_lines.extend([blank + l for l in func_code.split('\n')])

            # update code scope
            code_scope['numba'] = import_module('numba')

            # add the delay push in the end of the main code
            if not func_name.startswith('_npbrain_delayed_') and len(delay_keys) > 0:
                var2idx = self.ST['_var2idx']
                for st_k in func_delay_keys:
                    right = f'{self.name}_ST[{var2idx[st_k]}]'
                    left = f"{self.name}_ST[{var2idx['_' + st_k + '_offset']} + {self.name}_din]"
                    code_lines.append(f'{left} = {right}')
            code_lines.append('\n')

            # function comment
            func_name_stripped = tools.get_func_name(func, replace=True)
            code_lines.insert(0, f'# "{func_name_stripped}" step function of {self.name}')

            # update code scope
            for k in list(code_scope.keys()):
                if k in self._pars_to_update:
                    code_scope[k] = self._pars_to_update[k]
                if k in all_heter_pars:
                    all_heter_pars.remove(k)

            # the final results
            self._codegen[func_name_stripped] = {'scopes': code_scope, 'args': code_args,
                                                 'arg2calls': code_arg2call, 'codes': code_lines}

        # WARNING: heterogeneous parameter may not in the main step functions
        if len(all_heter_pars) > 0:
            raise ModelDefError(f'Heterogeneous parameters "{list(all_heter_pars)}" are not defined '
                                f'in main step function. NumpyBrain cannot recognize. Please check.')

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
                self.__step_mode_nb_single()

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
                if profile._auto_pep8:
                    func_code = autopep8.fix_code(func_code)
                exec(compile(func_code, '', 'exec'), code_scope)
                self.input_step = code_scope['input_step']

                # format function call
                code_arg2call = [code_arg2call[arg] for arg in code_args]
                func_call = f'{self.name}.input_step({", ".join(code_arg2call)})'

                if profile._show_formatted_code:
                    print(func_code)
                    print()
                    tools.show_code_scope(code_scope, ['__builtins__', 'input_step'])
                    print()
            else:
                self.input_step = None
                func_call = ''

            self._codegen['input'] = {'func': self.input_step, 'call': func_call}

        else:
            code_lines.append('\n')
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
            key = key.replace(',', '_')
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
                if profile._auto_pep8:
                    func_code = autopep8.fix_code(func_code)
                exec(compile(func_code, '', 'exec'), code_scope)
                self.monitor_step = code_scope['monitor_step']

                # format function call
                code_arg2call = [code_arg2call[arg] for arg in code_args]
                func_call = f'{self.name}.monitor_step({", ".join(code_arg2call)})'

                if profile._show_formatted_code:
                    tools.show_code_str(func_code)
                    tools.show_code_scope(code_scope, ('__builtins__', 'monitor_step'))
            else:
                self.monitor_step = None
                func_call = ''

            self._codegen['monitor'] = {'func': self.monitor_step, 'call': func_call}

        else:
            code_lines.append('\n')
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

        elif profile.is_numba_bk():  # non-numpy mode
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
            if profile._auto_pep8:
                func_code = autopep8.fix_code(func_code)
            exec(compile(func_code, '', 'exec'), code_scopes)

            self.merge_func = tools.jit(code_scopes['merge_func'])
            func_call = f'{self.name}.merge_func({", ".join(arg2calls_list)})'
            codes_of_calls.append(func_call)

            if profile._show_formatted_code:
                tools.show_code_str(func_code)
                tools.show_code_scope(code_scopes, ('__builtins__', 'merge_func'))

        else:
            raise NotImplementedError

        return codes_of_calls

    def _is_state_attr(self, arg):
        try:
            attr = getattr(self, arg)
        except AssertionError:
            raise ModelUseError(f'"{self.model.name}" need "{arg}", but it isn\'t defined in this model.')
        if self._cls_type == _NEU_GROUP:
            return isinstance(attr, NeuState)
        elif self._cls_type == _SYN_CONN:
            return isinstance(attr, SynState)
        else:
            raise ValueError

    def set_ST(self, new_ST):
        type_checker = self.model.requires['ST']
        try:
            type_checker.check(new_ST)
        except TypeMismatchError:
            raise ModelUseError(f'"new_ST" doesn\'t satisfy TypeChecker "{str(type_checker)}".')
        super(BaseEnsemble, self).__setattr__('ST', new_ST)

    def update_param(self, **kwargs):
        for k, v in kwargs.items():
            val_size = np.size(v)
            if val_size != 1:
                if val_size != self.num:
                    raise ModelUseError(f'The size of parameter "{k}" is wrong, "{val_size}" != 1 '
                                        f'and "{val_size}" != {self.num}.')
                else:
                    if self.model.vector_based:
                        if k not in self.model.heter_params_replace:
                            continue
                    self._hetero_pars[k] = v
            self._pars_to_update[k] = v

    @property
    def requires(self):
        return self.model.requires

    @property
    def _keywords(self):
        kws = [
            # attributes
            'model', 'num', 'ST', '_mon_vars',
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
