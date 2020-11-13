# -*- coding: utf-8 -*-

import inspect
import time
from copy import deepcopy

import autopep8

from .constants import ARG_KEYWORDS
from .constants import INPUT_OPERATIONS
from .constants import _NEU_GROUP
from .constants import _SYN_CONN
from .runner import Runner
from .types import NeuState
from .types import SynState
from .types import TypeChecker
from .types import TypeMismatchError
from .. import numpy as np
from .. import profile
from .. import tools
from ..errors import ModelDefError
from ..errors import ModelUseError

__all__ = [
    'BaseType',
    'BaseEnsemble',
    'ParsUpdate',
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

    def __init__(self,
                 requires,
                 steps,
                 name,
                 vector_based=True,
                 heter_params_replace=None):
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
        self.steps = []
        self.step_names = []
        self.step_scopes = dict()
        self.step_args = set()
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
            # function arg
            for arg in inspect.getfullargspec(func).args:
                if arg in ARG_KEYWORDS:
                    continue
                self.step_args.add(arg)
            # function scope
            scope = tools.get_func_scope(func, include_dispatcher=True)
            for k, v in scope.items():
                if k in self.step_scopes:
                    if v != self.step_scopes[k]:
                        raise ModelDefError(f'Find scope variable {k} have different values in '
                                            f'{self.name}: {k} = {v} and {k} = {self.step_scopes[k]}.\n'
                                            f'This maybe cause a grievous mistake in the future. Please change!')
                self.step_scopes[k] = v
            # function
            self.steps.append(func)
            # set attribute
            setattr(self, func_name, func)
        self.step_args = list(self.step_args)

        # integrators
        # -----------
        self.integrators = []
        for step in self.steps:
            self.integrators.extend(tools.find_integrators(step))
        self.integrators = list(set(self.integrators))

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
        for arg in self.step_args:
            if arg not in self.requires:
                warn = f'"{self.name}" requires "{arg}" as argument, but "{arg}" isn\'t declared in "requires".'
                warnings.append(warn)
        if len(warnings):
            print('\n'.join(warnings) + '\n')

        # delay keys
        self._delay_keys = {}

    def __str__(self):
        return f'{self.name}'


class ParsUpdate(dict):
    """Class for parameter updating.

    Structure of ``ParsUpdate``

    - origins : original parameters
    - num : number of the neurons
    - updates : parameters to update
    - heters : parameters to update, and they are heterogeneous
    - model : the model which this ParsUpdate belongs to

    """
    def __init__(self,
                 all_pars,
                 num,
                 model):
        assert isinstance(all_pars, dict)
        assert isinstance(num, int)

        super(ParsUpdate, self).__init__(origins=all_pars,
                                         num=num,
                                         heters=dict(),
                                         updates=dict(),
                                         model=model)

    def __setitem__(self, key, value):
        if profile.is_numpy_bk():
            raise ModelUseError('NumPy mode do not support modify parameters. '
                                'Please update parameters at the initialization of NeuType/SynType.')

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

    def __dir__(self):
        return str(self.all)

    def keys(self):
        """All parameters can be updated.

        Returns
        -------
        keys : list
            List of parameter names.
        """
        return self.origins.keys()

    def items(self):
        """All parameters, including keys and values.

        Returns
        -------
        items : iterable
            The iterable parameter items.
        """
        return self.all.items()

    def get(self, item):
        """Get the parameter value by its key.

        Parameters
        ----------
        item : str
            Parameter name.

        Returns
        -------
        value : any
            Parameter value.
        """
        return self.all.__getitem__(item)

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

    def __init__(self,
                 name,
                 num,
                 model,
                 monitors,
                 pars_update,
                 cls_type):
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
        self.pars = ParsUpdate(all_pars=model.step_scopes, num=num, model=model)
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

        # runner
        # -------
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

    def _is_state_attr(self, arg):
        try:
            attr = getattr(self, arg)
        except AttributeError:
            return False
        if self._cls_type == _NEU_GROUP:
            return isinstance(attr, NeuState)
        elif self._cls_type == _SYN_CONN:
            return isinstance(attr, SynState)
        else:
            raise ValueError

    def _build(self, mode, inputs=None, mon_length=0):
        # prerequisite
        self._type_checking()

        # results
        results = dict()

        # inputs
        if inputs:
            r = self.runner.format_input_code(inputs, mode=mode)
            results.update(r)

        # monitors
        if len(self._mon_vars):
            mon, r = self.runner.format_monitor_code(self._mon_vars, run_length=mon_length, mode=mode)
            results.update(r)
            self.mon.clear()
            self.mon.update(mon)

        # steps
        r = self.runner.format_step_codes(mode=mode)
        results.update(r)

        # merge
        calls = self.runner.merge_steps(results, mode=mode)

        if self._cls_type == _SYN_CONN:
            index_update_items = set()
            for func in self.model.steps:
                for arg in inspect.getfullargspec(func).args:
                    if self._is_state_attr(arg):
                        index_update_items.add(arg)
            for arg in index_update_items:
                calls.append(f'{self.name}.{arg}._update_delay_indices()')

        return calls

    @property
    def requires(self):
        return self.model.requires

    @property
    def _keywords(self):
        kws = [
            'model', 'num', '_mon_vars', 'mon', '_cls_type', '_keywords',
        ]
        if hasattr(self, 'model'):
            kws += self.model.step_names
        return kws

    def run(self, duration, inputs=(), report=False, report_percent=0.1):
        # times
        # ------
        if isinstance(duration, (int, float)):
            start, end = 0, duration
        elif isinstance(duration, (tuple, list)):
            assert len(duration) == 2, 'Only support duration with the format of "(start, end)".'
            start, end = duration
        else:
            raise ValueError(f'Unknown duration type: {type(duration)}')
        dt = profile.get_dt()
        times = np.arange(start, end, dt)
        times = np.asarray(times, dtype=np.float_)
        run_length = times.shape[0]

        # check inputs
        # -------------
        try:
            assert isinstance(inputs, (tuple, list))
        except AssertionError:
            raise ModelUseError('"inputs" must be a tuple/list.')
        if len(inputs) and not isinstance(inputs[0], (list, tuple)):
            if isinstance(inputs[0], str):
                inputs = [inputs]
            else:
                raise ModelUseError('Unknown input structure, only support inputs '
                                    'with format of "(key, value, [operation])".')
        for inp in inputs:
            try:
                assert 2 <= len(inp) <= 3
            except AssertionError:
                raise ModelUseError('For each target, you must specify "(key, value, [operation])".')
            if len(inp) == 3:
                try:
                    assert inp[2] in INPUT_OPERATIONS
                except AssertionError:
                    raise ModelUseError(f'Input operation only support '
                                        f'"{list(INPUT_OPERATIONS.keys())}", not "{inp[2]}".')

        # format inputs
        # -------------
        formatted_inputs = []
        for inp in inputs:
            # key
            try:
                assert isinstance(inp[0], str)
            except AssertionError:
                raise ModelUseError('For each input, input[0] must be a string '
                                    'to specify variable of the target.')
            key = inp[0]

            # value and data type
            if isinstance(inp[1], (int, float)):
                val = inp[1]
                data_type = 'fix'
            elif isinstance(inp[1], np.ndarray):
                val = inp[1]
                if val.shape[0] == run_length:
                    data_type = 'iter'
                else:
                    data_type = 'fix'
            else:
                raise ModelUseError('For each input, input[1] must be a numerical value to specify input values.')

            # operation
            if len(inp) == 3:
                ops = inp[2]
            else:
                ops = '+'

            format_inp = (key, val, ops, data_type)
            formatted_inputs.append(format_inp)

        # get step function
        # -------------------
        lines_of_call = self._build(mode=profile.get_backend(),
                                    inputs=formatted_inputs,
                                    mon_length=run_length)
        code_lines = ['def step_func(_t_, _i_, _dt_):']
        code_lines.extend(lines_of_call)
        code_scopes = {self.name: self}
        func_code = '\n  '.join(code_lines)
        if profile._auto_pep8:
            func_code = autopep8.fix_code(func_code)
        exec(compile(func_code, '', 'exec'), code_scopes)
        step_func = code_scopes['step_func']
        if profile._show_formatted_code:
            tools.show_code_str(func_code)
            tools.show_code_scope(code_scopes, ['__builtins__', 'step_func'])

        # run the model
        # -------------
        if report:
            t0 = time.time()
            step_func(_t_=times[0], _i_=0, _dt_=dt)
            print('Compilation used {:.4f} ms.'.format(time.time() - t0))

            print("Start running ...")
            report_gap = int(run_length * report_percent)
            t0 = time.time()
            for run_idx in range(1, run_length):
                step_func(_t_=times[run_idx], _i_=run_idx, _dt_=dt)
                if (run_idx + 1) % report_gap == 0:
                    percent = (run_idx + 1) / run_length * 100
                    print('Run {:.1f}% used {:.3f} s.'.format(percent, time.time() - t0))
            print('Simulation is done in {:.3f} s.'.format(time.time() - t0))
        else:
            for run_idx in range(run_length):
                step_func(_t_=times[run_idx], _i_=run_idx, _dt_=dt)

        self.mon['ts'] = times

    def __setattr__(self, key, value):
        if key in self._keywords:
            if hasattr(self, key):
                raise KeyError(f'"{key}" is a keyword in "{self._cls_type}" model, please change another name.')
        super(BaseEnsemble, self).__setattr__(key, value)
