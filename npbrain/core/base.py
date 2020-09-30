# -*- coding: utf-8 -*-

import inspect
from copy import deepcopy

from .. import profile
from .types import TypeChecker
from .types import ObjState
from .. import _numpy as np
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

    def __str__(self):
        return f'{self.name}'


class BaseEnsemble(object):

    def __init__(self, model, name, num, pars_update, vars_init, monitors):
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

        # parameters
        # ----------
        pars_update = dict() if pars_update is None else pars_update
        assert isinstance(pars_update, dict), '"pars_update" must be a dict.'
        parameters = deepcopy(self.model.parameters)
        for k, v in pars_update:
            val_size = np.size(v)
            if val_size != 1:
                if val_size != num:
                    raise ValueError(f'The size of parameter "{k}" is wrong, "{val_size}" != 1 '
                                     f'and "{val_size}" != {num}.')
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

        # step functions
        # ---------------
        if profile.is_numpy_bk():
            func_return = self.model.create_func(**pars_update)
            step_func = func_return['step_func']
            if callable(step_func):
                step_func = [step_func, ]
            elif isinstance(step_func, (tuple, list)):
                step_func = list(step_func)
            else:
                raise ValueError('"step_func" must be a callable, or a list/tuple of callable functions.')
            self.step_func = step_func

        elif profile.is_numba_bk():
            raise NotImplementedError

        else:
            raise NotImplementedError

        for func in step_func:
            func_name = func.__name__
            setattr(self, func_name, func)

        # monitors
        # ---------
        self.mon = helper.Dict()
        self._mon_vars = monitors
        if monitors is not None:
            for k in monitors:
                self.mon[k] = np.empty((1, 1), dtype=np.float_)

    def _type_checking(self):
        # check attribute and its type
        for key, type_checker in self.model.attributes.items():
            if not hasattr(self, key):
                raise AttributeError(f'"{self.name}" doesn\'t have "{key}" attribute.')
            if not type_checker.check(getattr(self, key)):
                raise TypeError(f'"{self.name}.{key}" doesn\'t satisfy TypeChecker "{str(type_checker)}".')

        # get function arguments
        step_func_args = []
        for func in self.step_func:
            args = [arg for arg in inspect.getfullargspec(func).args if arg != 'self']
            step_func_args.extend(args)

        # check step function arguments
        for i, args in enumerate(step_func_args):
            for arg in args:
                # Or, check "not (arg in ['ST', 't', 'i', 'din', 'dout', 'self'])"
                if not (arg in ['t', 'i', 'self']) and not hasattr(self, arg):
                    raise AttributeError(f'Function "{self.step_func[i].__name__}" in "{self.name}" requires '
                                         f'"{arg}" as argument, but "{arg}" is not defined in this model.')

    def _add_monitor(self, run_length):
        code_scope = {self.name: self}
        code_args, code_arg2call = set(), {}
        code_lines = []

        idx_no = 0

        for key, indices in self._mon_vars:
            attr_item = key.split('.')

            # get the left side #
            if len(attr_item) == 1:
                item, attr = attr_item[0], ''

                # if "item" is a field of "ST"
                if item in self.ST:
                    attr = 'ST'
                    shape = self.ST[key].shape

                    if profile.is_numpy_bk():
                        if indices is None:
                            line = f'{self.name}.mon["{key}"][i] = self.ST["{item}"]'
                        else:
                            idx_name = f'{self.name}_idx{idx_no}_ST_{item}'
                            line = f'{self.name}.mon["{key}"][i][{idx_name}] = self.ST["{item}"][{idx_name}]'
                            idx_no += 1
                            code_scope[idx_name] = indices
                    else:
                        idx = self.ST['_var2idx'][item]
                        mon_name = f'{self.name}_mon_ST_{item}'
                        target_name = f'{self.name}_ST'
                        if indices is None:
                            line = f'{mon_name}[i] = {self.name}_ST[{idx}]'
                        else:
                            idx_name = f'{self.name}_idx{idx_no}_ST_{item}'
                            line = f'for _mi_, _ti_ in enumerate({idx_name}): ' \
                                   f'{mon_name}[i, _mi_] = {self.name}_ST[{idx}, _ti_]'
                            idx_no += 1
                            code_scope[idx_name] = indices
                        code_args.add(mon_name)
                        code_arg2call[mon_name] = f'{self.name}.mon["{key}"]'
                        code_args.add(target_name)
                        code_arg2call[target_name] = f'{self.name}.ST["_data"]'

                # if "item" is the model attribute
                else:
                    attr, item = item, ''
                    assert hasattr(self, attr), f'Model "{self.name}" doesn\'t have "{attr}" attribute", ' \
                                                f'and "{self.name}.ST" doesn\'t have "{attr}" field.'
                    assert isinstance(getattr(self, attr), np.ndarray), f'NumpyBrain only support monitor of arrays.'

                    if profile.is_numpy_bk():
                        if indices is None:
                            pass

                        else:
                            pass

                    else:
                        if indices is None:
                            pass
                        else:
                            pass

            elif len(attr_item) == 2:
                attr, item = attr_item[0], attr_item[1]
                assert item in getattr(self, attr), f'"{self.name}.{attr}" doesn\'t have "{item}" field.'

                if profile.is_numpy_bk():
                    if indices is None:
                        pass

                    else:
                        pass

                else:
                    if indices is None:
                        pass
                    else:
                        pass

            else:
                raise ValueError(f'Unknown target : {key}.')

            self.mon[key] = np.zeros((run_length,) + shape, dtype=np.float_)

        #     if len(vs) == 1:
        #         assert k in self.ST, f'"{k}" isn\'t in ST.'
        #         shape = self.ST[k].shape
        #         code = 'self.mon[k][i] = self.ST[k]'
        #     elif len(vs) == 2:
        #         attr_name, k = vs
        #         try:
        #             attr = getattr(self, attr_name)
        #         except AttributeError:
        #             raise AttributeError(f'"{self.name}" does\'t have "{attr_name}" attribute.')
        #         assert isinstance(attr, ObjState), f'We can only monitor the field of "ObjState".'
        #         assert k in attr, f'"{k}" isn\'t in "{attr_name}".'
        #         shape = attr_name[k].shape
        #         code = f'self.mon[k][i] = self.{attr_name}[k]'
        #     else:
        #         raise KeyError(f'Unknown variable "{k}" to monitor.')
        #
        #     self.mon[k] = np.zeros((run_length,) + shape, dtype=np.float_)
        #     mon_codes.append(code)
        # print('\n\t'.join(mon_codes))
        # print(mon_codes)

    def _merge_step_func(self, run_length):
        self._add_monitor(run_length)

        if profile.is_numpy_bk():
            pass

        else:
            raise NotImplementedError

    def _get_step_calls(self, run_length):
        step_scope = dict()
        step_codes = []
        for func in self.step_func:
            func_name = func.__name__
            args = inspect.getfullargspec(func).args
            arg_code = ''
            for arg in args:
                if arg == 'self':
                    pass
                elif arg in ['t', 'i']:
                    arg_code += f'{args}, '
                else:
                    if isinstance(getattr(self, arg), ObjState):
                        arg_code += f'{self.name}.{arg}["_data"], '
                    else:
                        arg_code += f'{self.name}.{arg}, '
            code = f'{self.name}.{func_name}({arg_code[:-2]})'
            step_codes.append(code)
        step_scope[self.name] = self

        return step_codes, step_scope

    def change_ST(self, new_ST):
        type_checker = self.model.attributes['ST']
        if not type_checker.check(new_ST):
            raise TypeError(f'"new_ST" doesn\'t satisfy TypeChecker "{str(type_checker)}".')
        super(BaseEnsemble, self).__setattr__('ST', new_ST)
