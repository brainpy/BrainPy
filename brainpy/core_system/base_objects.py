# -*- coding: utf-8 -*-

import inspect
import re
from importlib import import_module

from copy import deepcopy
import autopep8

from .errors import ModelUseError
from .errors import ModelDefError
from .constants import ARG_KEYWORDS
from .constants import _NEU_GROUP
from .constants import _SYN_CONN
from .runner import Runner
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
    'BaseType',
    'BaseEnsemble',
]


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
            # function name
            func_name = tools.get_func_name(func, replace=True)
            self.step_names.append(func_name)
            # function
            self.steps.append(func)
            # function scope
            scope = tools.get_func_scope(func, include_dispatcher=True)
            for k, v in scope.items():
                if k in self.steps_scope:
                    if v != self.steps_scope[k]:
                        raise ModelDefError(f'Find scope variable {k} have different values in '
                                            f'{self.name}: {k} = {v} and {k} = {self.steps_scope[k]}.\n'
                                            f'This maybe cause a grievous mistake in the future. Please change!')
                self.steps_scope[k] = v
            # set attribute
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
                if arg in ARG_KEYWORDS:
                    continue
                if arg not in self.requires:
                    warn = f'"{self.name}" requires "{arg}" as argument, but "{arg}" isn\'t declared in "requires".'
                    warnings.append(warn)
        if len(warnings):
            print('\n'.join(warnings) + '\n')

        # delay keys
        self._delay_keys = set()

    def __str__(self):
        return f'{self.name}'


class ParsUpdate(dict):
    def __init__(self, all_pars, num, model):
        assert isinstance(all_pars, (tuple, list))
        assert isinstance(num, int)

        # structure of the ParsUpdate #
        # --------------------------- #
        # origins : original parameters
        # num : number of the neurons
        # heters : heterogeneous parameters
        # updates : parameters to update
        # model : the model belongs to

        super(ParsUpdate, self).__init__(origins=all_pars, num=num,
                                         heters=dict(), updates=dict(),
                                         model=model)

    def __setitem__(self, key, value):
        # check the existence of "key"
        if key not in self.origins:
            raise ModelUseError(f'Parameter "{key}" may be not defined in "{self.model.name}" variable scope.\n'
                                f'Or, "{key}" is used to compute an intermediate variable, and is not '
                                f'directly used by the step functions.')

        # check value size
        val_size = np.size(value)
        if val_size != 1:
            if val_size != self.num:
                raise ModelUseError(f'The size of parameter "{key}" is wrong, "{val_size}" != 1 '
                                    f'and "{val_size}" != {self.num}.')
            if np.size(self.origins[key]) != val_size:  # maybe default value is a heterogeneous value
                self.heters[key] = value

        # update
        self.updates[key] = value

    @property
    def origins(self):
        return self.__getitem__('origins')

    @property
    def heters(self):
        return self.__getitem__('heters')

    @property
    def updates(self):
        return self.__getitem__('updates')

    @property
    def num(self):
        return self.__getitem__('num')

    @property
    def model(self):
        return self.__getitem__('model')

    @property
    def all(self):
        origins = deepcopy(self.origins)
        origins.update(self.updates)
        return origins


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
        self.pars = ParsUpdate(all_pars=model.steps_scope, num=num, model=model)
        pars_update = dict() if pars_update is None else pars_update
        try:
            assert isinstance(pars_update, dict)
        except AssertionError:
            raise ModelUseError('"pars_update" must be a dict.')
        for k, v in pars_update.items():
            self.pars[k] = v

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

        #
        self.runner = Runner(ensemble=self)

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
                if not (arg in ARG_KEYWORDS + ['self']) and not hasattr(self, arg):
                    raise ModelUseError(f'Function "{self.model.step_names[i]}" in "{self.model.name}" '
                                        f'requires "{arg}" as argument, but "{arg}" is not defined in "{self.name}".')

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
        except AttributeError:
            raise ModelUseError(f'"{self.model.name}" need "{arg}", but it isn\'t defined in this model.')
        if self._cls_type == _NEU_GROUP:
            return isinstance(attr, NeuState)
        elif self._cls_type == _SYN_CONN:
            return isinstance(attr, SynState)
        else:
            raise ValueError

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
