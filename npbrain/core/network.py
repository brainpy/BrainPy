# -*- coding: utf-8 -*-

import time

import numpy as np

from npbrain.utils import profile
from npbrain.core.monitor import SpikeMonitor, StateMonitor, Monitor
from npbrain.core.neuron import Neurons
from npbrain.core.synapse import Synapses
from npbrain.utils.helper import Dict

__all__ = [
    'Network',
]


class Network(object):
    """The main simulation controller in ``NumpyBrain``.

    ``Network`` handles the running of a simulation. It contains a set of
    objects that are added with `add()`. The `run()` method
    actually runs the simulation. The main loop runs according to user add
    orders. The objects in the `Network` are accessible via their names, e.g.
    `net.name` would return the `object` (including neurons, synapses, and
    monitors) with this name.

    """

    def __init__(self, *args, **kwargs):
        # store and neurons and synapses
        self.neurons = []
        self.synapses = []
        self.monitors = []

        # store all objects
        self._objsets = Dict()
        self.objects = []

        # store states of objects and synapses
        self._neuron_states = None
        self._synapse_states = None

        # record the current step
        self.current_time = 0.

        # add objects
        self.add(*args, **kwargs)

    def add(self, *args, **kwargs):
        """Add object (neurons or synapses or monitor) to the network.

        Parameters
        ----------
        args : list, tuple
            The nameless objects.
        kwargs : dict
            The named objects, which can be accessed by `net.xxx`
            (xxx is the name of the object).
        """
        keywords = ['neurons', 'synapses', 'monitors', '_objsets', '_neuron_states',
                    '_synapse_states', 'current_time', 'add', 'run', 'run_time']
        for obj in args:
            self._add_obj(obj)
        for name, obj in kwargs.items():
            self._add_obj(obj)
            self._objsets.unique_add(name, obj)
            if name in keywords:
                raise ValueError('Invalid name: ', name)
            setattr(self, name, obj)

    def run(self, duration, report=False, inputs=(), repeat=False):
        """Run the simulation for the given duration.

        This function provides the most convenient way to run the network.
        For example:

        >>> # first of all, define the network we want.
        >>> import npbrain as nn
        >>> lif1 = nn.LIF(10, noise=0.2)
        >>> lif2 = nn.LIF(10, noise=0.5)
        >>> syn = nn.VoltageJumpSynapse(lif1, lif2, 1.0, nn.conn.all2all(lif1.num, lif2.num))
        >>> net = Network(syn, lif1, lif2)
        >>> # next, run the network.
        >>> net.run(100.) # run 100. ms
        >>>
        >>> # if you want to provide input to `lif1`
        >>> # for example, a constant input `11` (more complex inputs please use `input_factory.py`)
        >>> net.run(100., inputs=(lif1, 11.))
        >>>
        >>> # if you want to provide input to `lif1` in the period of 30-50 ms.
        >>> net.run(100., inputs=(lif1, 11., (30., 50.)))
        >>>
        >>> # moreover, if you want to provide input to `lif1` in the period of 30-50 ms,
        >>> # and provide input to `lif2` in the period of 10-100 ms.
        >>> net.run(100., inputs=[(lif1, 11., (30., 50.)), (lif2, -1., (10., 100.))])
        >>>
        >>> # if you want to known the running status in real-time.
        >>> net.run(100., report=True)
        >>>

        Parameters
        ----------
        duration : int, float
            The amount of simulation time to run for.
        report : bool
            Report the progress of the simulation.
        repeat : bool
            Whether repeat run this model. If `repeat=True`, every time
            call this method will initialize the object state.
        inputs : list, tuple
            The receivers, external inputs and durations.
        """
        # 1. checking
        # ------------
        self._check_run_order()

        # 2. initialization
        # ------------------

        # time
        dt = profile.get_dt()
        ts = np.arange(self.current_time, self.current_time + duration, dt)
        run_length = len(ts)

        # monitors
        for mon in self.monitors:
            mon.init_state(run_length)

        # neurons
        if repeat:
            if self._neuron_states is None:
                self._neuron_states = [neu.state.copy() for neu in self.neurons]
            else:
                for neu, state in zip(self.neurons, self._neuron_states):
                    neu.state[:] = state.copy()

        # synapses
        if repeat:
            if self._synapse_states is None:
                self._synapse_states = []
                for syn in self.synapses:
                    state = tuple(st.copy() for st in syn.state)
                    self._synapse_states.append([state, syn.var2index_array.copy()])
            else:
                for syn, (state, var_index) in zip(self.synapses, self._synapse_states):
                    for i, st in enumerate(state):
                        syn.state[i][:] = st.copy()
                    syn.var2index_array[:] = var_index.copy()

        # 3. format external inputs
        # --------------------------
        iterable_inputs, fixed_inputs, no_inputs = self._format_inputs_and_receiver(inputs, duration)

        # 4. run
        # ---------

        # initialize
        if report:
            t0 = time.time()
        self._input(0, iterable_inputs, fixed_inputs, no_inputs)
        self._step(t=ts[0], run_idx=0)

        # record time
        if report:
            print('Compilation used {:.4f} ms.'.format(time.time() - t0))
            print("Start running ...")
            report_gap = int(run_length / 10)
            t0 = time.time()

        # run
        for run_idx in range(1, run_length):
            t = ts[run_idx]
            self._input(run_idx, iterable_inputs, fixed_inputs, no_inputs)
            self._step(t=t, run_idx=run_idx)
            if report and ((run_idx + 1) % report_gap == 0):
                percent = (run_idx + 1) / run_length * 100
                print('Run {:.1f}% using {:.3f} s.'.format(percent, time.time() - t0))
        if report:
            print('Simulation is done. ')

        # 5. Finally
        # -----------
        self.current_time = duration

    def run_time(self):
        """Get the time points of the network.

        Returns
        -------
        times : numpy.ndarray
            The running time-steps of the network.
        """
        return np.arange(0, self.current_time, profile.get_dt())

    def _add_obj(self, obj):
        if isinstance(obj, Neurons):
            self.neurons.append(obj)
        elif isinstance(obj, Synapses):
            self.synapses.append(obj)
        elif isinstance(obj, Monitor):
            self.monitors.append(obj)
        else:
            raise ValueError('Unknown object type: {}'.format(type(obj)))
        self.objects.append(obj)

    def _check_run_order(self):
        for obj in self.objects:
            if isinstance(obj, Synapses):
                syn_order = self.objects.index(obj)
                pre_neu_order = self.objects.index(obj.pre)
                post_neu_order = self.objects.index(obj.post)
                if syn_order > post_neu_order or syn_order > pre_neu_order:
                    raise ValueError('Synapse "{}" must run before than the '
                                     'pre-/post-synaptic neurons.'.format(obj))

    def _format_inputs_and_receiver(self, inputs, duration):
        dt = profile.get_dt()
        # format inputs and receivers
        if len(inputs) > 1 and not isinstance(inputs[0], (list, tuple)):
            if isinstance(inputs[0], Neurons):
                inputs = [inputs]
            else:
                raise ValueError('Unknown input structure.')
        # ---------------------
        # classify input types
        # ---------------------
        # 1. iterable inputs
        # 2. fixed inputs
        iterable_inputs = []
        fixed_inputs = []
        neuron_with_inputs = []
        for input_ in inputs:
            # get "receiver", "input", "duration"
            if len(input_) == 2:
                obj, Iext = input_
                dur = (0, duration)
            elif len(input_) == 3:
                obj, Iext, dur = input_
            else:
                raise ValueError
            err = 'You can assign inputs only for added object. "{}" is not in the network.'
            if isinstance(obj, str):
                try:
                    obj = self._objsets[obj]
                except:
                    raise ValueError(err.format(obj))
            assert isinstance(obj, Neurons), "You can assign inputs only for Neurons."
            assert obj in self.objects, err.format(obj)
            assert len(dur) == 2, "Must provide the start and the end simulation time."
            assert 0 <= dur[0] < dur[1] <= duration
            dur = (int(dur[0] / dt), int(dur[1] / dt))
            neuron_with_inputs.append(obj)

            # judge the type of the inputs.
            if isinstance(Iext, (int, float)):
                Iext = np.ones(obj.num) * Iext
                fixed_inputs.append([obj, Iext, dur])
                continue
            size = np.shape(Iext)[0]
            run_length = dur[1] - dur[0]
            if size != run_length:
                if size == 1:
                    Iext = np.ones(obj.num) * Iext
                elif size == obj.num:
                    Iext = Iext
                else:
                    raise ValueError('Wrong size of inputs for', obj)
                fixed_inputs.append([obj, Iext, dur])
            else:
                input_size = np.size(Iext[0])
                err = 'The input size "{}" do not match with neuron ' \
                      'group size "{}".'.format(input_size, obj.num)
                assert input_size == 1 or input_size == obj.num, err
                iterable_inputs.append([obj, Iext, dur])
        # 3. no inputs
        no_inputs = []
        for neu in self.neurons:
            if neu not in neuron_with_inputs:
                no_inputs.append(neu)
        return iterable_inputs, fixed_inputs, no_inputs

    def _input(self, run_idx, iterable_inputs, fixed_inputs, no_inputs):
        # inputs
        for receiver, inputs, dur in iterable_inputs:
            if dur[0] <= run_idx <= dur[1]:
                obj_idx = run_idx - dur[0]
                receiver.state[-1] = inputs[obj_idx]
            else:
                receiver.state[-1] = 0.
        for receiver, inputs, dur in fixed_inputs:
            if dur[0] <= run_idx <= dur[1]:
                receiver.state[-1] = inputs
            else:
                receiver.state[-1] = 0.
        for receiver in no_inputs:
            receiver.state[-1] = 0.

    def _step(self, t, run_idx):
        for obj in self.objects:
            if isinstance(obj, Synapses):
                obj.collect_spike(obj.state, obj.pre.state, obj.post.state)
                obj.update_state(obj.state, t, obj.var2index_array)
                obj.output_synapse(obj.state, obj.var2index_array, obj.post.state, )
            elif isinstance(obj, Neurons):
                obj.update_state(obj.state, t)
            elif isinstance(obj, StateMonitor):
                vars_idx = obj.target_index_by_vars()
                obj.update_state(obj.target.state, obj.state, vars_idx, run_idx)
            elif isinstance(obj, SpikeMonitor):
                obj.update_state(obj.target.state, obj.time, obj.index, t)
            else:
                raise ValueError
