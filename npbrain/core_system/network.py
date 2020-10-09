# -*- coding: utf-8 -*-

import time
from pprint import pprint

import autopep8

from .base_objects import BaseEnsemble
from .base_objects import ModelUseError
from .neuron_group import NeuGroup
from .synapse_connection import SynConn
from .. import _numpy as np
from .. import profile

__all__ = [
    'Network',
]


class Network(object):
    """The main simulation controller in ``NumpyBrain``.

    ``Network`` handles the running of a simulation. It contains a set of
    objects that are added with `add()`. The `run()` method
    actually runs the simulation. The main loop runs according to user add
    orders. The objects in the `Network` are accessible via their names, e.g.
    `net.name` would return the `object` (including neurons and synapses).

    """

    def __init__(self, *args, **kwargs):
        # store and neurons and synapses
        self._all_neu_groups = []
        self._all_syn_conns = []
        self._all_objects = []

        # store all objects
        self._objsets = dict()

        # record the current step
        self._run_time = 0.

        # add objects
        self.add(*args, **kwargs)

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
        if name in self._keywords:
            raise ValueError(f'"{name}" is a keyword of "Network" class, please choose another name.')

        # add object in the network
        self._objsets[name] = obj
        setattr(self, name, obj)
        if obj.name != name:
            setattr(self, obj.name, obj)

    def add(self, *args, **kwargs):
        """Add object (neurons or synapses or monitor) to the network.

        Parameters
        ----------
        args : a_list, tuple
            The nameless objects.
        kwargs : dict
            The named objects, which can be accessed by `net.xxx`
            (xxx is the name of the object).
        """

        for obj in args:
            self._add_obj(obj)
        for name, obj in kwargs.items():
            self._add_obj(obj, name)

    def _get_step_function(self):
        code_lines = ['# network step function'
                      '\ndef step_func(_t_, _i_, _dt_):']
        code_scope = {}
        for obj in self._all_objects:
            code_lines.extend(obj._merge_steps())
            code_scope[obj.name] = obj

        func_code = '\n  '.join(code_lines)
        if profile.auto_pep8:
            func_code = autopep8.fix_code(func_code)
        exec(compile(func_code, '', 'exec'), code_scope)
        step_func = code_scope['step_func']

        if profile.show_codgen:
            print(func_code)
            print()
            code_scope.pop('__builtins__')
            code_scope.pop('step_func')
            pprint(code_scope)
            print()

        return step_func

    def _format_inputs(self, inputs, run_length):
        # check
        try:
            assert isinstance(inputs, (tuple, list))
        except AssertionError:
            raise ModelUseError('"inputs" must be a tuple/list.')
        if not isinstance(inputs[0], (list, tuple)):
            if isinstance(inputs[0], BaseEnsemble):
                inputs = [inputs]
            else:
                raise ModelUseError('Unknown input structure.')
        for inp in inputs:
            try:
                assert 3 <= len(inp) <= 4
            except AssertionError:
                raise ModelUseError('For each target, you must specify "(target, key, value, [operation])".')
            if len(inp) == 4:
                try:
                    assert inp[3] in ['+', '-', 'x', '/', '=']
                except AssertionError:
                    raise ModelUseError(f'Input operation only support "+, -, x, /, =", not "{inp[2]}".')

        # format input
        formatted_inputs = {}
        for inp in inputs:
            # target
            if isinstance(inp[0], str):
                target = getattr(self, inp[0]).name
            elif isinstance(inp[0], BaseEnsemble):
                target = inp[0].name
            else:
                raise KeyError(f'Unknown input target: {str(inp[0])}.')

            # key
            try:
                assert isinstance(inp[1], str)
            except AssertionError:
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
                raise ModelUseError('For each input, input[2] must be a numerical value to specify input values.')

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

        # first step
        for obj in self._all_objects:
            # type checking
            obj._type_checking()
            # step function
            obj._add_steps()
            # initialize monitor
            obj._add_monitor(run_length)

        # inputs
        format_inputs = self._format_inputs(inputs, run_length)
        for target, input in format_inputs.items():
            self._objsets[target]._add_input(input)

        # step function
        _step_func = self._get_step_function()

        return _step_func

    def run(self, duration, inputs=(), report=False, report_percent=0.1):
        """Run the simulation for the given duration.

        This function provides the most convenient way to run the network.
        For example:

        Parameters
        ----------
        duration : int, float
            The amount of simulation time to run for.
        inputs : a_list, tuple
            The receivers, external inputs and durations.
        report : bool
            Report the progress of the simulation.
        report_percent : float
            The speed to report simulation progress.
        """
        # initialization
        # ------------------
        ts = np.arange(self._run_time, self._run_time + duration, self.dt)
        ts = np.asarray(ts, dtype=profile.ftype)
        run_length = ts.shape[0]

        # 1. build
        # ----------
        _step_func = self.build(run_length, inputs)

        # 2. run
        # ---------
        dt = self.dt
        if report:
            t0 = time.time()
            _step_func(_t_=ts[0], _i_=0, _dt_=dt)
            print('Compilation used {:.4f} ms.'.format(time.time() - t0))

            print("Start running ...")
            report_gap = int(run_length * report_percent)
            t0 = time.time()
            for run_idx in range(1, run_length):
                _step_func(_t_=ts[run_idx], _i_=run_idx, _dt_=dt)
                if (run_idx + 1) % report_gap == 0:
                    percent = (run_idx + 1) / run_length * 100
                    print('Run {:.1f}% using {:.3f} s.'.format(percent, time.time() - t0))
            print('Simulation is done in {:.3f} s.'.format(time.time() - t0))
        else:
            for run_idx in range(run_length):
                _step_func(_t_=ts[run_idx], _i_=run_idx, _dt_=dt)

        # 3. Finally
        # -----------
        self._run_time = duration

    @property
    def _keywords(self):
        return [
            # attributes
            '_all_neu_groups', '_all_syn_conns', '_objsets', '_all_objects',
            '_run_time',
            # self functions
            'add', 'run', '_add_obj',
            # property
            'ts', '_keywords', 'dt',
        ]

    @property
    def ts(self):
        """Get the time points of the network.

        Returns
        -------
        times : numpy.ndarray
            The running time-steps of the network.
        """
        return np.arange(0, self._run_time, self.dt)

    @property
    def dt(self):
        return profile.get_dt()
