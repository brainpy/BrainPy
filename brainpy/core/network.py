# -*- coding: utf-8 -*-

import time
from collections import OrderedDict

import numpy as np
from numba import cuda

from . import base
from . import constants
from . import neurons
from . import utils
from .. import errors
from .. import profile

__all__ = [
    'Network',
]


class Network(object):
    """The main simulation controller in ``BrainPy``.

    ``Network`` handles the running of a simulation. It contains a set of
    objects that are added with `add()`. The `run()` method
    actually runs the simulation. The main loop runs according to user add
    orders. The objects in the `Network` are accessible via their names, e.g.
    `net.name` would return the `object` (including neurons and synapses).
    """

    def __init__(self, *args, mode=None, **kwargs):
        # record the current step
        self.t_start = 0.
        self.t_end = 0.

        # store all objects
        self._all_objects = OrderedDict()
        self.add(*args, **kwargs)

        # store the step function
        self._step_func = None

        if isinstance(mode, str):
            print('The "repeat" mode of the network is set to the default. '
                  'After version 0.4.0, "mode" setting will be removed.')

    def _add_obj(self, obj, name=None):
        # 1. check object type
        if not isinstance(obj, neurons.Ensemble):
            raise ValueError(f'Unknown object type "{type(obj)}". Network '
                             f'only supports NeuGroup and SynConn.')
        # 2. check object name
        name = obj.name if name is None else name
        if name in self._all_objects:
            raise KeyError(f'Name "{name}" has been used in the network, '
                           f'please change another name.')
        self._all_objects[name] = obj
        # 3. add object to the network
        setattr(self, name, obj)
        if obj.name != name:
            setattr(self, obj.name, obj)

    def add(self, *args, **kwargs):
        """Add object (neurons or synapses) to the network.

        Parameters
        ----------
        args
            The nameless objects.
        kwargs
            The named objects, which can be accessed by `net.xxx`
            (xxx is the name of the object).
        """

        for obj in args:
            self._add_obj(obj)
        for name, obj in kwargs.items():
            self._add_obj(obj, name)

    def format_inputs(self, inputs, run_length):
        """Format the user defined inputs.

        Parameters
        ----------
        inputs : tuple
            The inputs.
        run_length : int
            The running length.

        Returns
        -------
        formatted_input : dict
            The formatted input.
        """

        # 1. format the inputs to standard
        #    formats and check the inputs
        if not isinstance(inputs, (tuple, list)):
            raise errors.ModelUseError('"inputs" must be a tuple/list.')
        if len(inputs) > 0 and not isinstance(inputs[0], (list, tuple)):
            if isinstance(inputs[0], base.Ensemble):
                inputs = [inputs]
            else:
                raise errors.ModelUseError(
                    'Unknown input structure. Only supports "(target, key, value, [operation])".')
        for inp in inputs:
            if not 3 <= len(inp) <= 4:
                raise errors.ModelUseError('For each target, you must specify "(target, key, value, [operation])".')
            if len(inp) == 4:
                if inp[3] not in constants.INPUT_OPERATIONS:
                    raise errors.ModelUseError(f'Input operation only support '
                                               f'"{list(constants.INPUT_OPERATIONS.keys())}", '
                                               f'not "{inp[3]}".')

        # 2. format inputs
        formatted_inputs = {}
        for inp in inputs:
            # target
            if isinstance(inp[0], str):
                target = getattr(self, inp[0]).name
            elif isinstance(inp[0], base.Ensemble):
                target = inp[0].name
            else:
                raise KeyError(f'Unknown input target: {str(inp[0])}')

            # key
            if not isinstance(inp[1], str):
                raise errors.ModelUseError('For each input, input[1] must be a string '
                                           'to specify variable of the target.')
            key = inp[1]

            # value and data type
            if isinstance(inp[2], (int, float)):
                val = inp[2]
                data_type = 'fix'
            elif isinstance(inp[2], np.ndarray):
                val = inp[2]
                if val.shape[0] == run_length:
                    data_type = 'iter'
                else:
                    data_type = 'fix'
            else:
                raise errors.ModelUseError(f'For each input, input[2] must be a numerical value to '
                                           f'specify input values, but we get a {type(inp)}')

            # operation
            if len(inp) == 4:
                ops = inp[3]
            else:
                ops = '+'

            # final result
            if target not in formatted_inputs:
                formatted_inputs[target] = []
            format_inp = (key, val, ops, data_type)
            formatted_inputs[target].append(format_inp)
        return formatted_inputs

    def build(self, run_length, inputs=()):
        """Build the network.

        Parameters
        ----------
        run_length : int
            The running length.
        inputs : tuple, list
            The user-defined inputs.

        Returns
        -------
        step_func : callable
            The step function.
        """
        if not isinstance(run_length, int):
            raise errors.ModelUseError(f'The running length must be an int, but we get {run_length}')

        # inputs
        format_inputs = self.format_inputs(inputs, run_length)

        # codes for step function
        code_scopes = {}
        code_lines = ['# network step function\ndef step_func(_t, _i, _dt):']
        for obj in self._all_objects.values():
            if profile.run_on_gpu():
                if obj.model.mode != constants.SCALAR_MODE:
                    raise errors.ModelUseError(f'GPU mode only support scalar-based mode. '
                                               f'But {obj.model} is a {obj.model.mode}-based model.')
            code_scopes[obj.name] = obj
            code_scopes[f'{obj.name}_runner'] = obj.runner
            lines_of_call = obj.build(inputs=format_inputs.get(obj.name, None), mon_length=run_length)
            code_lines.extend(lines_of_call)
        if profile.run_on_gpu():
            code_scopes['cuda'] = cuda
        func_code = '\n  '.join(code_lines)

        # compile the step function
        exec(compile(func_code, '', 'exec'), code_scopes)
        step_func = code_scopes['step_func']

        # show
        if profile.show_format_code():
            utils.show_code_str(func_code.replace('def ', f'def network_'))
        if profile.show_code_scope():
            utils.show_code_scope(code_scopes, ['__builtins__', 'step_func'])

        return step_func

    def run(self, duration, inputs=(), report=False, report_percent=0.1,
            data_to_host=False, verbose=True):
        """Run the simulation for the given duration.

        This function provides the most convenient way to run the network.
        For example:

        Parameters
        ----------
        duration : int, float, tuple, list
            The amount of simulation time to run for.
        inputs : list, tuple
            The receivers, external inputs and durations.
        report : bool
            Report the progress of the simulation.
        report_percent : float
            The speed to report simulation progress.
        data_to_host : bool
            Transfer the gpu data to cpu. Available in CUDA backend.
        verbose : bool
            Show the error information.
        """
        # check the duration
        # ------------------
        if isinstance(duration, (int, float)):
            start, end = 0, duration
        elif isinstance(duration, (tuple, list)):
            if len(duration) != 2:
                raise errors.ModelUseError('Only support duration with the format of "(start, end)".')
            start, end = duration
        else:
            raise ValueError(f'Unknown duration type: {type(duration)}')
        self.t_start, self.t_end = start, end
        dt = profile.get_dt()
        ts = np.asarray(np.arange(start, end, dt), dtype=np.float_)
        run_length = ts.shape[0]

        if self._step_func is None:
            # initialize the function
            # -----------------------
            self._step_func = self.build(run_length, inputs)
        else:
            # check and reset inputs
            # ----------------------
            input_keep_same = True
            formatted_inputs = self.format_inputs(inputs, run_length)
            for obj in self._all_objects.values():
                obj_name = obj.name
                obj_inputs = obj.runner._inputs
                onj_input_keys = list(obj_inputs.keys())
                if obj_name in formatted_inputs:
                    current_inputs = formatted_inputs[obj_name]
                else:
                    current_inputs = []
                for key, val, ops, data_type in current_inputs:
                    if np.shape(obj_inputs[key][0]) != np.shape(val):
                        if verbose:
                            print(f'The current "{key}" input shape {np.shape(val)} is different '
                                  f'from the last input shape {np.shape(obj_inputs[key][0])}.')
                            input_keep_same = False
                    if obj_inputs[key][1] != ops:
                        if verbose:
                            print(f'The current "{key}" input operation "{ops}" is different '
                                  f'from the last operation "{obj_inputs[key][1]}".')
                            input_keep_same = False
                    obj.runner.set_data(f'{key.replace(".", "_")}_inp', val)
                    if key in onj_input_keys:
                        onj_input_keys.remove(key)
                    else:
                        input_keep_same = False
                        if verbose:
                            print(f'The input to a new key "{key}" in {obj_name}.')
                if len(onj_input_keys):
                    input_keep_same = False
                    if verbose:
                        print(f'The inputs of {onj_input_keys} in {obj_name} are not provided.')
            if input_keep_same:
                # reset monitors
                # --------------
                for obj in self._all_objects.values():
                    obj.reshape_mon(run_length)
            else:
                if verbose:
                    print('The network will be rebuild.')
                self._step_func = self.build(run_length, inputs)

        dt = self.dt
        if report:
            # Run the model with progress report
            # ----------------------------------
            t0 = time.time()
            self._step_func(_t=ts[0], _i=0, _dt=dt)
            print('Compilation used {:.4f} s.'.format(time.time() - t0))

            print("Start running ...")
            report_gap = int(run_length * report_percent)
            t0 = time.time()
            for run_idx in range(1, run_length):
                self._step_func(_t=ts[run_idx], _i=run_idx, _dt=dt)
                if (run_idx + 1) % report_gap == 0:
                    percent = (run_idx + 1) / run_length * 100
                    print('Run {:.1f}% used {:.3f} s.'.format(percent, time.time() - t0))
            print('Simulation is done in {:.3f} s.'.format(time.time() - t0))
        else:
            # Run the model
            # -------------
            for run_idx in range(run_length):
                self._step_func(_t=ts[run_idx], _i=run_idx, _dt=dt)

        # format monitor
        # --------------
        for obj in self._all_objects.values():
            obj.mon['ts'] = self.ts
            if data_to_host and profile.run_on_gpu():
                obj.runner.gpu_data_to_cpu()

    @property
    def ts(self):
        """Get the time points of the network.
        """
        return np.array(np.arange(self.t_start, self.t_end, self.dt), dtype=np.float_)

    @property
    def dt(self):
        return profile.get_dt()
