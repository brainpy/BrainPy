# -*- coding: utf-8 -*-

import inspect
from copy import deepcopy

from .types import TypeChecker
from .. import _numpy as np
from .. import profile
from ..utils import helper

__all__ = [
    # errors
    'ModelDefError',

    # base types
    'BaseType',
    'BaseEnsemble',
]


class ModelDefError(Exception):
    """Model definition error."""
    pass


_neu_no = 0
_syn_no = 0


class BaseType(object):
    """The base type of neuron and synapse.

    Structure of a BaseType instantiation:

    - parameters = a_dict  (model parameters)
    - variables = a_dict   (model variables)
    - attributes = a_dict  (essential attributes for model running)
     - a1 = av1
     - a2 = av2
     - ...
    - step_func = a_list  (collection of the step functions)
     - f1 = callable
     - f2 = callable
     - ...

    Parameters
    ----------
    create_func : callable
        The function to create the model.
    name : str, optional
        Model name.
    group_based : bool
        Whether the model is written in the neuron-group level or in the single-neuron level.
    type_ : str
        Whether the model is a 'neuron' or a 'synapse' model.
    """

    def __init__(self, create_func, name=None, group_based=True, type_='neu'):
        # type : neuron based or group based code
        # ---------------------------------------
        self.group_based = group_based

        # name
        # -----
        if name is None:
            if type_ == 'neu':
                global _neu_no
                self.name = f'NeuType{_neu_no}'
                _neu_no += 1
            elif type_ == 'syn':
                global _syn_no
                self.name = f'SynType{_syn_no}'
                _syn_no += 1
            else:
                raise KeyError('Unknown group type: ', type_)
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
        assert 'attrs' in func_return, \
            '"attrs" (specify variables the model need) must be defined in the return.'
        assert 'step_func' in func_return, \
            '"step_func" (step functions at each time step) must be defined in the return.'
        self.create_func = create_func
        self.func_return = func_return

        # parameters
        # -----------
        parameters = inspect.getcallargs(create_func)
        self.parameters = dict(parameters)

        # attributes
        # -----------
        attributes = func_return['attrs']
        assert isinstance(attributes, dict), '"attrs" only supports dict.'
        assert 'ST' in attributes, '"ST" must be defined in "attrs".'
        self.attributes = attributes
        for k, v in attributes.items():
            assert isinstance(v, TypeChecker), f'The value of "{k}" in "attrs" must be a TypeChecker.'
            setattr(self, k, v)

        # variables
        # ----------
        self.variables = self.attributes['ST']._vars

        # step functions
        # --------------
        step_func = func_return['step_func']
        if callable(step_func):
            step_func = [step_func]
        elif isinstance(step_func, (list, tuple)):
            step_func = list(step_func)
        else:
            raise ValueError('"step_func" must be a callable, or a list/tuple of callable functions.')
        self.step_func, self.step_names = [], []
        for func in step_func:
            assert callable(func), '"step_func" must be a list/tuple of callable functions.'
            func_name = func.__name__
            self.step_names.append(func_name)
            self.step_func.append(func)
            setattr(self, func_name, func)

        # check consistence between function
        # arguments and model attributes
        # ----------------------------------
        warnings = []
        for func in self.step_func:
            for arg in inspect.getfullargspec(func).args:
                if arg in ['t', 'i']:
                    continue
                if arg not in self.attributes:
                    warn = f'"{self.name}" requires "{arg}" as argument, but "{arg}" isn\'t declared in "attrs".'
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
        assert cls_type in ['neu_group', 'syn_conn'], 'Only support "neu_group" and "syn_conn".'
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
        assert isinstance(vars_init, dict), '"vars_init" must be a dict.'
        variables = deepcopy(self.model.variables)
        for k, v in vars_init:
            if k not in self.model.variables:
                raise KeyError(f'variable "{k}" is not defined in "{self.model.name}".')
            variables[k] = v
        self.vars_init = variables

        # parameters
        # ----------
        hetero_pars = {}
        pars_update = dict() if pars_update is None else pars_update
        assert isinstance(pars_update, dict), '"pars_update" must be a dict.'
        parameters = deepcopy(self.model.parameters)
        for k, v in pars_update:
            val_size = np.size(v)
            if val_size != 1:
                if val_size != num:
                    raise ValueError(f'The size of parameter "{k}" is wrong, "{val_size}" != 1 '
                                     f'and "{val_size}" != {num}.')
                else:
                    hetero_pars[k] = v
            parameters[k] = v
        self.pars_update = parameters
        if profile.is_numba_bk():
            import numba as nb
            max_size = max([np.size(v) for v in parameters.values()])
            if max_size > 1:
                self.PA = nb.typed.Dict.empty(key_type=nb.types.unicode_type, value_type=nb.types.float_[:])
                for k, v in parameters.items():
                    self.PA[k] = np.ones(self.num, dtype=np.float_) * v
            else:
                self.PA = nb.typed.Dict.empty(key_type=nb.types.unicode_type, value_type=nb.types.float_)
                for k, v in parameters.items():
                    self.PA[k] = v
        else:
            self.PA = parameters

        # step functions
        # ---------------
        if not model.group_based:
            if len(hetero_pars) > 0:
                raise ValueError('Single neuron/synapse model doesn\'t support heterogeneous parameters.')
            if self._cls_type == 'syn_conn' and (self.pre_group is None or self.post_group is None):
                raise ValueError('Single synapse model must provide "pre_group" and "post_group".')

        if profile.is_numpy_bk():
            self.step_func = self._get_steps_from_model(self.pars_update)

        elif profile.is_numba_bk():
            raise NotImplementedError

        else:
            raise NotImplementedError

        # monitors
        # ---------
        self.mon = helper.DictPlus()
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
                        raise ValueError(f'Unknown monitor item: {str(var)}')
            elif isinstance(monitors, dict):
                for k, v in monitors.items():
                    self._mon_vars.append((k, v))
                    self.mon[k] = np.empty((1, 1), dtype=np.float_)
            else:
                raise TypeError(f'Unknown monitors type: {type(monitors)}')

        # code generation results
        # -----------------------
        self._codegen = dict()

        # model update schedule
        # ---------------------
        self._schedule = ['input'] + self.model.step_names + ['monitor']

    def _get_steps_from_model(self, pars_update):
        func_return = self.model.create_func(**pars_update)
        step_func = func_return['step_func']
        if callable(step_func):
            step_func = [step_func, ]
        elif isinstance(step_func, (tuple, list)):
            step_func = list(step_func)
        else:
            raise ValueError('"step_func" must be a callable, or a list/tuple of callable functions.')
        return step_func

    def _type_checking(self):
        # check attribute and its type
        for key, type_checker in self.model.attributes.items():
            if not hasattr(self, key):
                raise AttributeError(f'"{self.name}" doesn\'t have "{key}" attribute.')
            if not type_checker.check(getattr(self, key)):
                raise TypeError(f'"{self.name}.{key}" doesn\'t satisfy TypeChecker "{str(type_checker)}".')

        # get function arguments
        for i, func in enumerate(self.step_func):
            for arg in inspect.getfullargspec(func).args:
                if not (arg in ['t', 'i', 'self']) and not hasattr(self, arg):
                    raise AttributeError(f'Function "{self.step_func[i].__name__}" in "{self.model.name}" requires '
                                         f'"{arg}" as argument, but "{arg}" is not defined in "{self.name}".')

    def _add_steps(self):
        if profile.is_numpy_bk():
            for func in self.step_func:
                func_name = func.__name__
                func_args = inspect.getfullargspec(func).args
                if self.model.group_based:
                    setattr(self, func_name, func)
                    arg_calls = []
                    for arg in func_args:
                        if arg == 'self': pass
                        elif arg in ['t', 'i']: arg_calls.append(arg)
                        else: arg_calls.append(f"{self.name}.{arg}")
                    func_call = f'{self.name}.{func_name}({", ".join(arg_calls)})'
                    self._codegen[func_name] = {'func': func, 'call': func_call}

                else:
                    code_lines = [f'def {func_name}({", ".join(func_args)}):']
                    code_lines.append(f'  for ni in range(self.pre_group.num):')
                    code_lines.append(f'    for si in self.:')
                    arg_calls = []
                    for arg in func_args:
                        if arg == 'self':
                            pass
                        elif arg in ['t', 'i']:
                            arg_calls.append(arg)
                        else:
                            arg_calls.append(f"{self.name}.{arg}")

                    func_call = f'{self.name}.{func_name}({", ".join(arg_calls)})'

                    self._codegen[func_name] = {'func': func, 'call': func_call}

        elif profile.is_numba_bk():
            code_scope, code_args, code_arg2call, code_lines = {self.name: self}, set(), {}, []

            for func in self.step_func:
                pass

        else:
            raise NotImplementedError

    def _add_input(self, key_val_ops_types):
        code_scope, code_args, code_arg2call, code_lines = {self.name: self}, set(), {}, []
        input_idx = 0

        # check datatype of the input
        # ----------------------------
        has_iter = False
        for _, _, _, t in key_val_ops_types:
            assert t in ['iter', 'fix'], 'Only support inputs of "iter" and "fix" types.'
            if t == 'iter':
                has_iter = True
        if has_iter:
            code_args.add('i')
            code_arg2call['i'] = 'i'

        # check data operations
        # ----------------------
        for _, _, ops, _ in key_val_ops_types:
            assert ops in ['-', '+', 'x', '/', '='], 'Only support five operations: +, -, x, /, ='
        ops2str = {'-': 'sub', '+': 'add', 'x': 'mul', '/': 'div', '=': 'assign'}

        # generate code of input function
        # --------------------------------
        for key, val, ops, data_type in key_val_ops_types:
            attr_item = key.split('.')

            # get the left side #
            if len(attr_item) == 1 and (attr_item[0] not in self.ST):  # if "item" is the model attribute
                attr, item = attr_item[0], ''
                assert hasattr(self, attr), f'Model "{self.name}" doesn\'t have "{attr}" attribute", ' \
                                            f'and "{self.name}.ST" doesn\'t have "{attr}" field.'
                assert isinstance(getattr(self, attr), np.ndarray), f'NumpyBrain only support input to arrays.'

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
                    raise ValueError(f'Unknown target : {key}.')
                assert item in getattr(self, attr), f'"{self.name}.{attr}" doesn\'t have "{item}" field.'

                if profile.is_numpy_bk():
                    left = f'{self.name}.{attr}["{item}"]'
                else:
                    idx = getattr(self, attr)['_var2idx'][item]
                    left = f'{self.name}_{attr}{idx}]'
                    code_args.add(f'{self.name}_{attr}')
                    code_arg2call[f'{self.name}_{attr}'] = f'{self.name}.{attr}["_data"]'

            # get the right side #
            right = f'{self.name}_input{input_idx}_{attr}_{item}_{ops2str[ops]}'
            code_scope[right] = val
            if data_type == 'iter':
                right = right + '[i]'
            input_idx += 1

            # final code line #
            if ops == '=':
                code_lines.append(left + " = " + right)
            else:
                code_lines.append(left + f" {ops}= " + right)

        # final code
        # ----------
        code_lines.insert(0, f'# Input step function of {self.name}')

        if profile.is_numpy_bk():
            code_args = list(code_args)
            code_lines.insert(0, f'\ndef input_step({", ".join(code_args)}):')

            # compile function
            exec(compile('\n  '.join(code_lines), '', 'exec'), code_scope)
            self.input_step = code_scope['input_step']

            # format function call
            code_arg2call = [code_arg2call[arg] for arg in code_args]
            func_call = f'{self.name}.input_step({", ".join(code_arg2call)})'

            if profile.show_codgen:
                print("\n" + '\n\t'.join(code_lines))
                print("\n" + func_call)

            self._codegen['input'] = {'func': self.input_step, 'call': func_call}

        else:
            self._codegen['input'] = {'scopes': code_scope, 'args': code_args,
                                      'arg2calls': code_arg2call, 'codes': code_lines}

    def _add_monitor(self, run_length):
        code_scope, code_args, code_arg2call, code_lines = {self.name: self}, set(('i',)), {'i': 'i'}, []
        idx_no = 0

        # generate code of monitor function
        # ---------------------------------
        for key, indices in self._mon_vars:
            # check indices #
            if indices is not None:
                if isinstance(indices, list):
                    assert isinstance(indices[0], int), 'Monitor index only supports list [int] or 1D array.'
                elif isinstance(indices, np.ndarray):
                    assert np.ndim(indices) == 1, 'Monitor index only supports list [int] or 1D array.'
                else:
                    raise ValueError(f'Unknown monitor index type: {type(indices)}.')

            attr_item = key.split('.')

            # get the code line #
            if (len(attr_item) == 1) and (attr_item[0] not in getattr(self, 'ST')):
                attr = attr_item[0]
                assert hasattr(self, attr), f'Model "{self.name}" doesn\'t have "{attr}" attribute", ' \
                                            f'and "{self.name}.ST" doesn\'t have "{attr}" field.'
                assert isinstance(getattr(self, attr), np.ndarray), \
                    f'NumpyBrain only support monitor of arrays.'
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
                        line = f'{mon_name}[i] = {target_name}'
                    else:
                        line = f'{mon_name}[i] = {target_name}[{idx_name}]'
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
                    raise ValueError(f'Unknown target : {key}.')

                shape = getattr(self, attr)[item].shape

                idx_name = f'{self.name}_idx{idx_no}_{attr}_{item}'
                if profile.is_numpy_bk():
                    if indices is None:
                        line = f'{self.name}.mon["{key}"][i] = {self.name}.{attr}["{item}"]'
                    else:
                        line = f'{self.name}.mon["{key}"][i] = {self.name}.{attr}["{item}"][{idx_name}]'
                        code_scope[idx_name] = indices
                        idx_no += 1
                else:
                    idx = getattr(self, attr)['_var2idx'][item]
                    mon_name = f'{self.name}_mon_{attr}_{item}'
                    target_name = f'{self.name}_{attr}'
                    if indices is None:
                        line = f'{mon_name}[i] = {target_name}[{idx}]'
                    else:
                        line = f'{mon_name}[i] = {target_name}[{idx}][{idx_name}]'
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
        code_lines.insert(0, f'# Monitor step function of {self.name}')

        if profile.is_numpy_bk():
            code_args = list(code_args)
            code_lines.insert(0, f'\ndef monitor_step({", ".join(code_args)}):')

            # compile function
            exec(compile('\n  '.join(code_lines), '', 'exec'), code_scope)
            self.monitor_step = code_scope['monitor_step']

            # format function call
            code_arg2call = [code_arg2call[arg] for arg in code_args]
            func_call = f'{self.name}.monitor_step({", ".join(code_arg2call)})'

            if profile.show_codgen:
                print("\n" + '\n\t'.join(code_lines))
                print("\n" + func_call)

            self._codegen['monitor'] = {'func': self.monitor_step, 'call': func_call}

        else:
            self._codegen['monitor'] = {'scopes': code_scope, 'args': code_args,
                                        'arg2calls': code_arg2call, 'codes': code_lines}

    def _merge_steps(self):
        codes_of_calls = []  # call the compiled functions
        if profile.is_numpy_bk():  # numpy mode
            for item in self._schedule:
                if item in self._codegen:
                    call = self._codegen[item]['call']
                    codes_of_calls.append(call)

        else:  # non-numpy mode
            lines, scopes, args, arg2calls = [], dict(), set(), dict()
            for item in self._schedule:
                if item in self._codegen:
                    lines.extend(self._codegen[item]['codes'])
                    scopes.update(self._codegen[item]['scopes'])
                    args = args | self._codegen[item]['args']
                    arg2calls.update(self._codegen[item]['arg2calls'])

            args = list(args)
            arg2calls_list = [arg2calls[arg] for arg in args]
            lines.insert(0, f'\ndef merge_func({", ".join(args)})')
            exec(compile('\n  '.join(lines), '', 'exec'), scopes)

            self.merge_func = scopes['merge_func']
            call = f'{self.name}.merge_func({", ".join(arg2calls_list)})'
            codes_of_calls.append(call)

            if profile.show_codgen:
                print("\n" + '\n\t'.join(lines))
                print("\n" + call)

        return codes_of_calls

    def set_ST(self, new_ST):
        type_checker = self.model.attributes['ST']
        if not type_checker.check(new_ST):
            raise TypeError(f'"new_ST" doesn\'t satisfy TypeChecker "{str(type_checker)}".')
        super(BaseEnsemble, self).__setattr__('ST', new_ST)

    @property
    def _keywords(self):
        kws = [
            # attributes
            'model', 'num', 'ST', 'PA', 'vars_init', 'pars_update', '_mon_vars',
            'mon', '_cls_type', '_codegen', '_keywords', 'step_func', '_schedule',
            # assigned functions
            'monitor_step', 'input_step', 'merge_func',
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
        assert isinstance(schedule, (list, tuple)), '"schedule" must be a list/tuple.'
        all_func_names = ['input', 'monitor'] + self.model.step_names

        for s in schedule:
            assert s in all_func_names, f'Unknown step function "{s}" for "{self._cls_type}" model.'
        super(BaseEnsemble, self).__setattr__('_schedule', schedule)

    def __setattr__(self, key, value):
        if key in self._keywords:
            if hasattr(self, key):
                raise KeyError(f'"{key}" is a keyword in "{self._cls_type}" model, please change another name.')
        super(BaseEnsemble, self).__setattr__(key, value)
