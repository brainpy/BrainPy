# -*- coding: utf-8 -*-

import inspect

from .types import TypeChecker
from .types import ObjState
from .. import _numpy as bnp

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
    - step_funcs = a_list  (collection of the step functions)
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
        assert 'steps' in func_return, \
            '"steps" (step functions at each time step) must be defined in the return.'
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
        step_funcs = func_return['step_funcs']
        if callable(step_funcs):
            step_funcs = [step_funcs]
        elif isinstance(step_funcs, (list, tuple)):
            step_funcs = list(step_funcs)
        else:
            raise ValueError('"step_funcs" must be a callable, or a list/tuple of callable functions.')
        self.step_funcs, self.step_names = [], []
        for func in step_funcs:
            assert callable(func), '"step_funcs" must be a list/tuple of callable functions.'
            func_name = func.__name__
            self.step_names.append(func_name)
            self.step_funcs.append(func)
            setattr(self, func_name, func)

    def __str__(self):
        return f'{self.name}'


class BaseEnsemble(object):
    __slots__ = ['model', '_mon_vars', 'mon', 'num', 'step_funcs', 'ST', 'name']

    def init_monitor(self, length):
        mon_codes = []
        mon_scope = {'self': self}
        for k in self._mon_vars:
            vs = k.split('.')
            if len(vs) == 1:
                assert k in self.ST, f'"{k}" isn\'t in ST.'
                shape = self.ST[k].shape
                code = 'self.mon[k][i] = self.ST[k]'
            elif len(vs) == 2:
                attr_name, k = vs
                try:
                    attr = getattr(self, attr_name)
                except AttributeError:
                    raise AttributeError(f'"{self.name}" does\'t have "{attr_name}" attribute.')
                assert isinstance(attr, ObjState), f'We can only monitor the field of "ObjState".'
                assert k in attr, f'"{k}" isn\'t in "{attr_name}".'
                shape = attr_name[k].shape
                code = f'self.mon[k][i] = self.{attr_name}[k]'
            else:
                raise KeyError(f'Unknown variable "{k}" to monitor.')

            self.mon[k] = bnp.zeros((length,) + shape, dtype=bnp.float_)
            mon_codes.append(code)
        print('\n\t'.join(mon_codes))
        print(mon_codes)

        return mon_scope, mon_codes

    def get_func_call(self):
        pass

    def type_checking(self):
        # check attribute and its type
        for key, type_checker in self.model.attributes.items():
            if not hasattr(self, key):
                raise AttributeError(f'"{self.name}" doesn\'t have "{key}" attribute.')
            if not type_checker.check(getattr(self, key)):
                raise TypeError(f'"{self.name}.{key}" doesn\'t satisfy TypeChecker "{str(type_checker)}".')

        # get function arguments
        step_func_args = []
        for func in self.step_funcs:
            args = [arg for arg in inspect.getfullargspec(func).args if arg != 'self']
            step_func_args.extend(args)

        # check step function arguments
        for i, args in enumerate(step_func_args):
            for arg in args:
                # if not (arg in ['ST', 't', 'i', 'din', 'dout', 'self']) and not hasattr(self, arg):
                if not (arg in ['t', 'i', 'self']) and not hasattr(self, arg):
                    raise AttributeError(f'Function "{self.step_funcs[i].__name__}" in "{self.name}" requires '
                                         f'"{arg}" as argument, but "{arg}" is not defined in this model.')

    def change_ST(self, new_ST):
        type_checker = self.model.attributes['ST']
        if not type_checker.check(new_ST):
            raise TypeError(f'"new_ST" doesn\'t satisfy TypeChecker "{str(type_checker)}".')
        super(BaseEnsemble, self).__setattr__('ST', new_ST)
