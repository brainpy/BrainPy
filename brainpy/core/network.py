# -*- coding: utf-8 -*-

import time

import numpy as np
from numba import cuda

from .base import Ensemble
from .constants import INPUT_OPERATIONS
from .constants import SCALAR_MODE
from .neurons import NeuGroup
from .synapses import SynConn
from .. import profile
from .. import tools
from ..errors import ModelUseError

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

    def __init__(self, *args, mode='once', **kwargs):
        # store and neurons and synapses
        self._all_neu_groups = []
        self._all_syn_conns = []
        self._all_objects = []

        # store all objects
        self._objsets = dict()

        # record the current step
        self.t_start = 0.
        self.t_end = 0.
        self.t_duration = 0.

        # add objects
        self.add(*args, **kwargs)

        # check
        assert mode in ['once', 'repeat']
        self.mode = mode

        self._step_func = None

    def _add_obj(self, obj, name=None):
        # check object type
        self._all_objects.append(obj)
        if isinstance(obj, NeuGroup):
            self._all_neu_groups.append(obj)
        elif isinstance(obj, SynConn):
            self._all_syn_conns.append(obj)
        else:
            raise ValueError(f'Unknown object type "{type(obj)}". Network only support NeuGroup and SynConn.')

        # check object name
        name = obj.name if name is None else name
        if name in self._objsets:
            raise KeyError(f'Name "{name}" has been used in the network, please change another name.')

        # add object in the network
        self._objsets[name] = obj
        setattr(self, name, obj)
        if obj.name != name:
            setattr(self, obj.name, obj)

    def add(self, *args, **kwargs):
        """Add object (neurons or synapses or monitor) to the network.

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

    def _format_inputs(self, inputs, run_length):
        # check
        try:
            assert isinstance(inputs, (tuple, list))
        except AssertionError:
            raise ModelUseError('"inputs" must be a tuple/list.')

        if len(inputs) > 0 and not isinstance(inputs[0], (list, tuple)):
            if isinstance(inputs[0], Ensemble):
                inputs = [inputs]
            else:
                raise ModelUseError('Unknown input structure.')
        for inp in inputs:
            if not 3 <= len(inp) <= 4:
                raise ModelUseError('For each target, you must specify "(target, key, value, [operation])".')
            if len(inp) == 4:
                if inp[3] not in INPUT_OPERATIONS:
                    raise ModelUseError(f'Input operation only support '
                                        f'"{list(INPUT_OPERATIONS.keys())}", not "{inp[3]}".')

        # format input
        formatted_inputs = {}
        for inp in inputs:
            # target
            if isinstance(inp[0], str):
                target = getattr(self, inp[0]).name
            elif isinstance(inp[0], Ensemble):
                target = inp[0].name
            else:
                raise KeyError(f'Unknown input target: {str(inp[0])}')

            # key
            if not isinstance(inp[1], str):
                raise ModelUseError('For each input, input[1] must be a string '
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
                raise ModelUseError(f'For each input, input[2] must be a numerical value to '
                                    f'specify input values, but we get a {type(inp)}')

            # operation
            if len(inp) == 4:
                ops = inp[3]
            else:
                ops = '+'

            # append
            if target not in formatted_inputs:
                formatted_inputs[target] = []
            format_inp = (key, val, ops, data_type)
            formatted_inputs[target].append(format_inp)

        return formatted_inputs

    def build(self, run_length, inputs=()):
        assert isinstance(run_length, int)
        code_scopes = {}
        code_lines = ['# network step function\n'
                      'def step_func(_t, _i, _dt):']

        # inputs
        format_inputs = self._format_inputs(inputs, run_length)

        for obj in self._all_objects:
            if profile.run_on_gpu():
                if obj.model.mode != SCALAR_MODE:
                    raise ModelUseError('GPU mode only support scalar-based mode.')

            code_scopes[obj.name] = obj
            code_scopes[f'{obj.name}_runner'] = obj.runner
            lines_of_call = obj._build(inputs=format_inputs.get(obj.name, None), mon_length=run_length)
            code_lines.extend(lines_of_call)
        if profile.run_on_gpu():
            code_scopes['cuda'] = cuda
        func_code = '\n  '.join(code_lines)
        exec(compile(func_code, '', 'exec'), code_scopes)
        step_func = code_scopes['step_func']

        if profile.show_format_code():
            tools.show_code_str(func_code.replace('def ', f'def network_'))
        if profile.show_format_code():
            tools.show_code_scope(code_scopes, ['__builtins__', 'step_func'])

        return step_func

    def run(self, duration, inputs=(), report=False, report_percent=0.1):
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
        """
        # check the duration
        # ------------------
        if isinstance(duration, (int, float)):
            start, end = 0, duration
        elif isinstance(duration, (tuple, list)):
            if len(duration) != 2:
                raise ModelUseError('Only support duration with the format of "(start, end)".')
            start, end = duration
        else:
            raise ValueError(f'Unknown duration type: {type(duration)}')
        self.t_start, self.t_end = start, end
        dt = profile.get_dt()
        ts = np.asarray(np.arange(start, end, dt), dtype=np.float_)
        run_length = ts.shape[0]

        if self.mode != 'repeat' or self._step_func is None:
            # initialize the function
            # -----------------------
            self._step_func = self.build(run_length, inputs)
            self.t_duration = end - start
        else:
            # check running duration
            # ----------------------
            if self.t_duration != (end - start):
                raise ModelUseError(f'Each run in "repeat" mode must be done '
                                    f'with the same duration, but got '
                                    f'{self.t_duration} != {end - start}.')

            # check and reset inputs
            # ----------------------
            formatted_inputs = self._format_inputs(inputs, run_length)
            for obj_name, inps in formatted_inputs.items():
                obj = getattr(self, obj_name)
                obj_inputs = obj.runner._inputs
                all_keys = list(obj_inputs.keys())
                for key, val, ops, data_type in inps:
                    if np.shape(obj_inputs[key][0]) != np.shape(val):
                        raise ModelUseError(f'The input shape for "{key}" should keep the same. '
                                            f'However, we got the last input shape '
                                            f'= {np.shape(obj_inputs[key][0])}, '
                                            f'and the current input shape = {np.shape(val)}')
                    if obj_inputs[key][1] != ops:
                        raise ModelUseError(f'The input operation for "{key}" should keep the same. '
                                            f'However, we got the last operation is '
                                            f'"{obj_inputs[key][1]}", and the current operation '
                                            f'is "{ops}"')
                    obj.runner.set_data(f'{key.replace(".", "_")}_inp', val)
                    all_keys.remove(key)
                if len(all_keys):
                    raise ModelUseError(f'The inputs of {all_keys} are not provided.')

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
        for obj in self._all_objects:
            if profile.run_on_gpu():
                obj.runner.gpu_data_to_cpu()
            obj.mon['ts'] = self.ts

    @property
    def ts(self):
        """Get the time points of the network.
        """
        return np.array(np.arange(self.t_start, self.t_end, self.dt), dtype=np.float_)

    @property
    def dt(self):
        return profile.get_dt()
