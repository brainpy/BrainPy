# -*- coding: utf-8 -*-

import inspect
import re
import time
from copy import deepcopy

import numpy as np
from numba import cuda

from . import constants
from . import runner
from . import types
from . import utils
from .. import errors
from .. import profile
from .. import tools

__all__ = [
    'ObjType',
    'Ensemble',
    'ParsUpdate',
]


class ObjType(object):
    """The base type of neuron and synapse.

    Parameters
    ----------
    name : str, optional
        Model name.
    """

    def __init__(self, ST, name, steps, requires=None, mode='vector', hand_overs=None, ):
        self.mode = mode
        self.name = name
        if not isinstance(ST, types.ObjState):
            raise errors.ModelDefError('"ST" must be an instance of ObjState.')
        self.ST = ST

        # requires
        # ---------
        if requires is None:
            requires = dict()
        if not isinstance(requires, dict):
            raise errors.ModelDefError('"requires" only supports dict.')
        self.requires = requires
        for k, v in requires.items():
            if isinstance(v, type):
                raise errors.ModelDefError(f'In "requires", you must instantiate '
                                           f'the type checker of "{k}". '
                                           f'Like "{v.__name__}()".')
            if not isinstance(v, types.TypeChecker):
                raise errors.ModelDefError(f'In "requires", each value must be a '
                                           f'{types.TypeChecker.__name__}, '
                                           f'but got "{type(v)}" for "{k}".')

        # steps
        # ------
        self.steps = []
        self.step_names = []
        self.step_scopes = dict()
        self.step_args = set()
        step_vars = set()
        if callable(steps):
            steps = [steps]
        elif isinstance(steps, (list, tuple)):
            steps = list(steps)
        else:
            raise errors.ModelDefError('"steps" must be a callable, or a '
                                       'list/tuple of callable functions.')
        for func in steps:
            if not callable(func):
                raise errors.ModelDefError('"steps" must be a list/tuple of callable functions.')

            # function name
            func_name = tools.get_func_name(func, replace=True)
            self.step_names.append(func_name)

            # function arg
            for arg in inspect.getfullargspec(func).args:
                if arg in constants.ARG_KEYWORDS:
                    continue
                self.step_args.add(arg)

            # function scope
            scope = utils.get_func_scope(func, include_dispatcher=True)
            for k, v in scope.items():
                if k in self.step_scopes:
                    if v != self.step_scopes[k]:
                        raise errors.ModelDefError(
                            f'Find scope variable {k} have different values in '
                            f'{self.name}: {k} = {v} and {k} = {self.step_scopes[k]}.\n'
                            f'This maybe cause a grievous mistake in the future. '
                            f'Please change!')
                self.step_scopes[k] = v

            # function
            self.steps.append(func)

            # set attribute
            setattr(self, func_name, func)

            # get the STATE variables
            step_vars.update(re.findall(r'ST\[[\'"](\w+)[\'"]\]', tools.get_main_code(func)))

        self.step_args = list(self.step_args)

        # variables
        # ----------
        self.variables = ST._vars
        for var in step_vars:
            if var not in self.variables:
                raise errors.ModelDefError(f'Variable "{var}" is used in {self.name}, '
                                           f'but not defined in "ST".')

        # integrators
        # -----------
        self.integrators = []
        for step in self.steps:
            self.integrators.extend(utils.find_integrators(step))
        self.integrators = list(set(self.integrators))

        # delay keys
        # ----------
        self._delay_keys = []

        # hand overs
        # ---------------
        if hand_overs is not None:
            if not isinstance(hand_overs, dict):
                raise errors.ModelUseError('"hand_overs" must be a dict.')
        else:
            hand_overs = dict()
        self.hand_overs = hand_overs

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

    def __init__(self, all_pars, num, model):
        assert isinstance(all_pars, dict)
        assert isinstance(num, int)

        super(ParsUpdate, self).__init__(origins=all_pars,
                                         num=num,
                                         heters=dict(),
                                         updates=dict(),
                                         model=model)

    def __setitem__(self, key, value):
        # check the existence of "key"
        if key not in self.origins:
            raise errors.ModelUseError(f'Parameter "{key}" may be not defined in '
                                       f'"{self.model.name}" variable scope.\n'
                                       f'Or, "{key}" is used to compute an '
                                       f'intermediate variable, and is not '
                                       f'directly used by the step functions.')

        # check value size
        val_size = np.size(value)
        if val_size != 1:
            if val_size != self.num:
                raise errors.ModelUseError(
                    f'The size of parameter "{key}" is wrong, "{val_size}" != 1 '
                    f'and "{val_size}" != {self.num}.')
            if np.size(self.origins[key]) != val_size:  # maybe default value is a heterogeneous value
                self.heters[key] = value

        # update
        if profile.run_on_cpu():
            self.updates[key] = value
        else:
            if isinstance(value, (int, float)):
                self.updates[key] = value
            elif value.__class__.__name__ == 'DeviceNDArray':
                self.updates[key] = value
            elif isinstance(value, np.ndarray):
                self.updates[key] = cuda.to_device(value)
            else:
                raise ValueError(f'GPU mode cannot support {type(value)}.')

    def __getitem__(self, item):
        if item in self.updates:
            return self.updates[item]
        elif item in self.origins:
            return self.origins[item]
        else:
            super(ParsUpdate, self).__getitem__(item)

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
        return super(ParsUpdate, self).__getitem__('origins')

    @property
    def heters(self):
        return super(ParsUpdate, self).__getitem__('heters')

    @property
    def updates(self):
        return super(ParsUpdate, self).__getitem__('updates')

    @property
    def num(self):
        return super(ParsUpdate, self).__getitem__('num')

    @property
    def model(self):
        return super(ParsUpdate, self).__getitem__('model')

    @property
    def all(self):
        origins = deepcopy(self.origins)
        origins.update(self.updates)
        return origins


class Ensemble(object):
    """Base Ensemble class.

    Parameters
    ----------
    name : str
        Name of the (neurons/synapses) ensemble.
    num : int
        The number of the neurons/synapses.
    model : ObjType
        The (neuron/synapse) model.
    monitors : list, tuple, None
        Variables to monitor.
    pars_update : dict, None
        Parameters to update.
    cls_type : str
        Class type.
    """

    def __init__(self, name, num, model, monitors, pars_update, cls_type, satisfies=None, ):
        # class type
        # -----------
        if not cls_type in [constants.NEU_GROUP_TYPE, constants.SYN_CONN_TYPE]:
            raise errors.ModelUseError(f'Only support "{constants.NEU_GROUP_TYPE}" '
                                       f'and "{constants.SYN_CONN_TYPE}".')
        self._cls_type = cls_type

        # model
        # -----
        self.model = model

        # name
        # ----
        self.name = name
        if not self.name.isidentifier():
            raise errors.ModelUseError(
                f'"{self.name}" isn\'t a valid identifier according to Python '
                f'language definition. Please choose another name.')

        # num
        # ---
        self.num = num

        # parameters
        # ----------
        self.pars = ParsUpdate(all_pars=model.step_scopes, num=num, model=model)
        pars_update = dict() if pars_update is None else pars_update
        if not isinstance(pars_update, dict):
            raise errors.ModelUseError('"pars_update" must be a dict.')
        for k, v in pars_update.items():
            self.pars[k] = v

        # monitors
        # ---------
        self.mon = tools.DictPlus()
        self._mon_items = []
        if monitors is not None:
            if isinstance(monitors, (list, tuple)):
                for var in monitors:
                    if isinstance(var, str):
                        self._mon_items.append((var, None))
                        self.mon[var] = np.empty((1, 1), dtype=np.float_)
                    elif isinstance(var, (tuple, list)):
                        self._mon_items.append((var[0], var[1]))
                        self.mon[var[0]] = np.empty((1, 1), dtype=np.float_)
                    else:
                        raise errors.ModelUseError(f'Unknown monitor item: {str(var)}')
            elif isinstance(monitors, dict):
                for k, v in monitors.items():
                    self._mon_items.append((k, v))
                    self.mon[k] = np.empty((1, 1), dtype=np.float_)
            else:
                raise errors.ModelUseError(f'Unknown monitors type: {type(monitors)}')

        # runner
        # -------
        self.runner = runner.Runner(ensemble=self)

        # hand overs
        # ----------
        # 1. attributes
        # 2. functions
        for attr_key, attr_val in model.hand_overs.items():
            setattr(self, attr_key, attr_val)

        # satisfies
        # ---------
        if satisfies is not None:
            if not isinstance(satisfies, dict):
                raise errors.ModelUseError('"satisfies" must be dict.')
            for key, val in satisfies.items():
                setattr(self, key, val)

    def _is_state_attr(self, arg):
        try:
            attr = getattr(self, arg)
        except AttributeError:
            return False
        if self._cls_type == constants.NEU_GROUP_TYPE:
            return isinstance(attr, types.NeuState)
        elif self._cls_type == constants.SYN_CONN_TYPE:
            return isinstance(attr, types.SynState)
        else:
            raise ValueError

    def type_checking(self):
        """Check the data type needed for step function.
        """
        # 1. check ST and its type
        if not hasattr(self, 'ST'):
            raise errors.ModelUseError(f'"{self.name}" doesn\'t have "ST" attribute.')
        try:
            self.model.ST.check(self.ST)
        except errors.TypeMismatchError:
            raise errors.ModelUseError(f'"{self.name}.ST" doesn\'t satisfy TypeChecker "{str(self.model.ST)}".')

        # 2. check requires and its type
        for key, type_checker in self.model.requires.items():
            if not hasattr(self, key):
                raise errors.ModelUseError(f'"{self.name}" doesn\'t have "{key}" attribute.')
            try:
                type_checker.check(getattr(self, key))
            except errors.TypeMismatchError:
                raise errors.ModelUseError(f'"{self.name}.{key}" doesn\'t satisfy TypeChecker "{str(type_checker)}".')

        # 3. check data (function arguments) needed
        for i, func in enumerate(self.model.steps):
            for arg in inspect.getfullargspec(func).args:
                if not (arg in constants.ARG_KEYWORDS + ['self']) and not hasattr(self, arg):
                    raise errors.ModelUseError(
                        f'Function "{self.model.step_names[i]}" in "{self.model.name}" '
                        f'requires "{arg}" as argument, but "{arg}" is not defined in "{self.name}".')

    def reshape_mon(self, run_length):
        for key, val in self.mon.items():
            if key == 'ts':
                continue
            shape = val.shape
            if run_length < shape[0]:
                self.mon[key] = val[:run_length]
            elif run_length > shape[0]:
                append = np.zeros((run_length - shape[0],) + shape[1:])
                self.mon[key] = np.vstack([val, append])
        if profile.run_on_gpu():
            for key, val in self.mon.items():
                key_gpu = f'mon_{key}_cuda'
                val_gpu = cuda.to_device(val)
                setattr(self.runner, key_gpu, val_gpu)
                self.runner.gpu_data[key_gpu] = val_gpu

    def build(self, inputs=None, mon_length=0):
        """Build the object for running.

        Parameters
        ----------
        inputs : list, tuple
            The object inputs.
        mon_length : int
            The monitor length.

        Returns
        -------
        calls : list, tuple
            The code lines to call step functions.
        """

        # 1. prerequisite
        # ---------------
        if profile.run_on_gpu():
            if self.model.mode != constants.SCALAR_MODE:
                raise errors.ModelUseError(f'GPU mode only support scalar-based mode. '
                                           f'But {self.model} is a {self.model.mode}-based model.')
        self.type_checking()

        # 2. Code results
        # ---------------
        code_results = dict()
        # inputs
        if inputs:
            r = self.runner.get_codes_of_input(inputs)
            code_results.update(r)
        # monitors
        if len(self._mon_items):
            mon, r = self.runner.get_codes_of_monitor(self._mon_items, run_length=mon_length)
            code_results.update(r)
            self.mon.clear()
            self.mon.update(mon)
        # steps
        r = self.runner.get_codes_of_steps()
        code_results.update(r)

        # 3. code calls
        # -------------
        calls = self.runner.merge_codes(code_results)
        if self._cls_type == constants.SYN_CONN_TYPE:
            if self.delay_len > 1:
                calls.append(f'{self.name}.ST._update_delay_indices()')

        return calls

    def run(self, duration, inputs=(), report=False, report_percent=0.1):
        """The running function.

        Parameters
        ----------
        duration : float, int, tuple, list
            The running duration.
        inputs : list, tuple
            The model inputs with the format of ``[(key, value [operation])]``.
        report : bool
            Whether report the running progress.
        report_percent : float
            The percent of progress to report.
        """

        # times
        # ------
        if isinstance(duration, (int, float)):
            start, end = 0., duration
        elif isinstance(duration, (tuple, list)):
            assert len(duration) == 2, 'Only support duration setting with the format of "(start, end)".'
            start, end = duration
        else:
            raise ValueError(f'Unknown duration type: {type(duration)}')
        times = np.asarray(np.arange(start, end, profile.get_dt()), dtype=np.float_)
        run_length = times.shape[0]

        # check inputs
        # -------------
        if not isinstance(inputs, (tuple, list)):
            raise errors.ModelUseError('"inputs" must be a tuple/list.')
        if len(inputs) and not isinstance(inputs[0], (list, tuple)):
            if isinstance(inputs[0], str):
                inputs = [inputs]
            else:
                raise errors.ModelUseError('Unknown input structure, only support inputs '
                                           'with format of "(key, value, [operation])".')
        for inp in inputs:
            if not 2 <= len(inp) <= 3:
                raise errors.ModelUseError('For each target, you must specify "(key, value, [operation])".')
            if len(inp) == 3 and inp[2] not in constants.INPUT_OPERATIONS:
                raise errors.ModelUseError(f'Input operation only supports '
                                           f'"{list(constants.INPUT_OPERATIONS.keys())}", '
                                           f'not "{inp[2]}".')

        # format inputs
        # -------------
        formatted_inputs = []
        for inp in inputs:
            # key
            if not isinstance(inp[0], str):
                raise errors.ModelUseError('For each input, input[0] must be a string '
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
                raise errors.ModelUseError('For each input, input[1] must be a '
                                           'numerical value to specify input values.')
            # operation
            if len(inp) == 3:
                ops = inp[2]
            else:
                ops = '+'
            # input
            format_inp = (key, val, ops, data_type)
            formatted_inputs.append(format_inp)

        # get step function
        # -------------------
        lines_of_call = self.build(inputs=formatted_inputs, mon_length=run_length)
        code_lines = ['def step_func(_t, _i, _dt):']
        code_lines.extend(lines_of_call)
        code_scopes = {self.name: self, f"{self.name}_runner": self.runner}
        if profile.run_on_gpu():
            code_scopes['cuda'] = cuda
        func_code = '\n  '.join(code_lines)
        exec(compile(func_code, '', 'exec'), code_scopes)
        step_func = code_scopes['step_func']
        if profile.show_format_code():
            utils.show_code_str(func_code)
        if profile.show_code_scope():
            utils.show_code_scope(code_scopes, ['__builtins__', 'step_func'])

        # run the model
        # -------------
        dt = profile.get_dt()
        if report:
            t0 = time.time()
            step_func(_t=times[0], _i=0, _dt=dt)
            print('Compilation used {:.4f} s.'.format(time.time() - t0))

            print("Start running ...")
            report_gap = int(run_length * report_percent)
            t0 = time.time()
            for run_idx in range(1, run_length):
                step_func(_t=times[run_idx], _i=run_idx, _dt=dt)
                if (run_idx + 1) % report_gap == 0:
                    percent = (run_idx + 1) / run_length * 100
                    print('Run {:.1f}% used {:.3f} s.'.format(percent, time.time() - t0))
            print('Simulation is done in {:.3f} s.'.format(time.time() - t0))
        else:
            for run_idx in range(run_length):
                step_func(_t=times[run_idx], _i=run_idx, _dt=dt)

        if profile.run_on_gpu():
            self.runner.gpu_data_to_cpu()
        self.mon['ts'] = times

    def get_schedule(self):
        """Get the schedule (running order) of the update functions.

        Returns
        -------
        schedule : list, tuple
            The running order of update functions.
        """
        return self.runner.get_schedule()

    def set_schedule(self, schedule):
        """Set the schedule (running order) of the update functions.

        For example, if the ``self.model`` has two step functions: `step1`, `step2`.
        Then, you can set the shedule by using:

        >>> set_schedule(['input', 'step1', 'step2', 'monitor'])
        """
        self.runner.set_schedule(schedule)

    @property
    def requires(self):
        return self.model.requires
