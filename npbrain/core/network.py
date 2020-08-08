# -*- coding: utf-8 -*-

import inspect
import time

import numpy as np

from .monitor import SpikeMonitor, StateMonitor
from .neuron import Neurons
from .synapse import Synapses
from ..utils import helper
from ..utils import profile

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
        self.state_monitors = []
        self.spike_monitors = []

        # store all objects
        self._objsets = helper.Dict()
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

    def run(self, duration, report=False, inputs=(), repeat=False, report_percent=0.1):
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

        # 2. initialization
        # ------------------

        # time
        dt = profile.get_dt()
        ts = np.arange(self.current_time, self.current_time + duration, dt)
        ts = np.asarray(ts, dtype=profile.ftype)
        run_length = len(ts)

        # monitors
        for mon in self.state_monitors:
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
                    state = (None if syn.state is None else syn.state.copy(), syn.delay_state.copy())
                    self._synapse_states.append(state)
            else:
                for syn, state in zip(self.synapses, self._synapse_states):
                    for i, (st, dst) in enumerate(state):
                        if st is not None:
                            syn.state[:] = st.copy()
                        syn.delay_state[:] = dst.copy()

        # generate step function
        # -------------------------

        def get_args(func):
            if hasattr(func, 'py_func'):
                args_ = inspect.getfullargspec(func.py_func)[0]
            else:
                args_ = inspect.getfullargspec(func)[0]
            return args_

        step_func_str = '''\ndef step_func(t, i):'''
        step_func_local = {}

        # synapse update function

        for oi, syn in enumerate(self.synapses):
            corresponds = {'syn_st': syn.state, 'delay_st': syn.delay_state,
                           'pre_st': syn.pre.state, 'post_st': syn.post.state,
                           'syn_state': syn.state, 'delay_state': syn.delay_state,
                           'pre_state': syn.pre.state, 'post_state': syn.post.state}
            # update_state()
            args = get_args(syn.update_state)
            step_func_str += '\n\t' + 'syn_update{}('.format(oi)
            step_func_local['syn_update{}'.format(oi)] = syn.update_state
            for arg in args:
                if arg in ['t', 'i']:
                    step_func_str += arg + ", "
                elif arg in ['delay_idx', 'in_idx']:
                    step_func_str += 'syn{}.var2index["g_in"], '.format(oi)
                    step_func_local['syn' + str(oi)] = syn
                else:
                    step_func_str += arg + str(oi) + ', '
                    step_func_local[arg + str(oi)] = corresponds[arg]
            step_func_str += ')'

            # output_synapse()
            args = get_args(syn.output_synapse)
            step_func_str += '\n\t' + 'syn_output{}('.format(oi)
            step_func_local['syn_output{}'.format(oi)] = syn.output_synapse
            for arg in args:
                if args in ['t', 'i']:
                    step_func_str += arg + ', '
                elif arg in ['output_idx', 'out_idx']:
                    step_func_str += 'syn{}.var2index["g_out"], '.format(oi)
                    step_func_local['syn' + str(oi)] = syn
                else:
                    step_func_str += arg + str(oi) + ', '
                    step_func_local[arg + str(oi)] = corresponds[arg]
            step_func_str += ')'

        # neuron update function
        for oi, neu in enumerate(self.neurons):
            step_func_str += '\n\t' + 'neu_update{}('.format(oi)
            step_func_local['neu_update{}'.format(oi)] = neu.update_state
            args = get_args(neu.update_state)
            for arg in args:
                if arg in ['t', 'i']:
                    step_func_str += arg + ', '
                else:
                    step_func_str += arg + str(oi) + ', '
                    step_func_local[arg + str(oi)] = neu.state
            step_func_str += ')'

        # state monitor function
        for oi, mon in enumerate(self.state_monitors):
            args = get_args(mon.update_state)
            if len(args) == 3:
                step_func_str += '\n\tst_mon_update{oi}(obj_st{oi}, mon_st{oi}, i)'.format(oi=oi)
                step_func_local['st_mon_update{}'.format(oi)] = mon.update_state
                step_func_local['obj_st{}'.format(oi)] = mon.target.state
                step_func_local['mon_st{}'.format(oi)] = mon.state
            elif len(args) == 5:
                step_func_str += '\n\tst_mon_update{oi}(target{oi}.delay_state, ' \
                                          'mon_st{oi}, ' \
                                          'target{oi}.var2index["g_out"], ' \
                                          'target{oi}.var2index["g_in"], ' \
                                          'i)'.format(oi=oi)
                step_func_local['st_mon_update{}'.format(oi)] = mon.update_state
                step_func_local['mon_st{}'.format(oi)] = mon.state
                step_func_local['target{}'.format(oi)] = mon.target
            elif len(args) == 6:
                step_func_str += '\n\tst_mon_update{oi}(target{oi}.state, ' \
                                          'target{oi}.delay_state, ' \
                                          'mon_st{oi}, ' \
                                          'target{oi}.var2index["g_out"], ' \
                                          'target{oi}.var2index["g_in"], ' \
                                          'i)'.format(oi=oi)
                step_func_local['st_mon_update{}'.format(oi)] = mon.update_state
                step_func_local['mon_st{}'.format(oi)] = mon.state
                step_func_local['target{}'.format(oi)] = mon.target
            else:
                raise ValueError('Unknown arguments of monitor update_state().')

        # spike monitor function
        for oi, mon in enumerate(self.spike_monitors):
            step_func_str += '\n\tsp_mon_update{oi}(obj_st{oi}, mon_t{oi}, mon_i{oi}, t)'.format(oi=oi)
            step_func_local['sp_mon_update{}'.format(oi)] = mon.update_state
            step_func_local['obj_st{}'.format(oi)] = mon.target.state
            step_func_local['mon_t{}'.format(oi)] = mon.time
            step_func_local['mon_i{}'.format(oi)] = mon.index

        # update `dalay_idx` and `output_idx`
        for oi, syn in enumerate(self.synapses):
            step_func_str += '\n\tsyn{oi}.var2index["g_in"] = (syn{oi}.var2index["g_in"] + 1) % ' \
                             'syn{oi}.delay_len'.format(oi=oi)
            step_func_str += '\n\tsyn{oi}.var2index["g_out"] = (syn{oi}.var2index["g_out"] + 1) % ' \
                             'syn{oi}.delay_len'.format(oi=oi)
            step_func_local['syn' + str(oi)] = syn

        if profile.debug:
            print('Step function :')
            print('----------------------------')
            print(step_func_str)
            print()
            from pprint import pprint
            pprint(step_func_local)
            print()
        exec(compile(step_func_str, '', 'exec'), step_func_local)
        self._step = step_func_local['step_func']

        # generate input function
        # -------------------------

        iterable_inputs, fixed_inputs, no_inputs = self._format_inputs_and_receiver(inputs, duration)

        input_func_local = {}
        neu_state_args = {}

        input_func_str = '''\ndef input_func(run_idx, {arg}):'''

        ni = 0
        for receiver, inputs, dur in iterable_inputs:
            input_func_str += '''
    if {du0} <= run_idx <= {du1}:
        obj_idx = run_idx - {du0}
        {neu_st}[-1] = {input}[obj_idx]
    else:
        {neu_st}[-1] = 0.
        '''.format(du0=dur[0], du1=dur[1], neu_st='neu_st' + str(ni), input='in' + str(ni))
            input_func_local['in' + str(ni)] = inputs
            neu_state_args['neu_st' + str(ni)] = receiver.state
            ni += 1

        for receiver, inputs, dur in fixed_inputs:
            input_func_str += '''
    if {du0} <= run_idx <= {du1}:
        {neu_st}[-1] = {input}
    else:
        {neu_st}[-1] = 0.
        '''.format(du0=dur[0], du1=dur[1], neu_st='neu_st' + str(ni), input='in' + str(ni))
            input_func_local['in' + str(ni)] = inputs
            neu_state_args['neu_st' + str(ni)] = receiver.state
            ni += 1

        for receiver in no_inputs:
            input_func_str += '''\n\t{neu_st}[-1] = 0.'''.format(neu_st='neu_st' + str(ni))
            neu_state_args['neu_st' + str(ni)] = receiver.state
            ni += 1

        input_func_str = input_func_str.format(arg=','.join(neu_state_args.keys()))

        if profile.debug:
            print('Input function :')
            print('----------------------------')
            print(input_func_str)

            pprint(input_func_local)
            print()
        exec(compile(input_func_str, '', 'exec'), input_func_local)
        self._input = helper.autojit(input_func_local['input_func'])

        # 4. run
        # ---------
        if report:
            t0 = time.time()
            self._input(run_idx=0, **neu_state_args)
            self._step(t=ts[0], i=0)
            print('Compilation used {:.4f} ms.'.format(time.time() - t0))

            print("Start running ...")
            report_gap = int(run_length * report_percent)
            t0 = time.time()
            for run_idx in range(1, run_length):
                self._input(run_idx=run_idx, **neu_state_args)
                self._step(t=ts[run_idx], i=run_idx)
                if (run_idx + 1) % report_gap == 0:
                    percent = (run_idx + 1) / run_length * 100
                    print('Run {:.1f}% using {:.3f} s.'.format(percent, time.time() - t0))
            print('Simulation is done. ')
        else:
            for run_idx in range(run_length):
                self._input(run_idx=run_idx, **neu_state_args)
                self._step(t=ts[run_idx], i=run_idx)

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
        elif isinstance(obj, StateMonitor):
            self.state_monitors.append(obj)
        elif isinstance(obj, SpikeMonitor):
            self.spike_monitors.append(obj)
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
                Iext = np.ones(obj.num, dtype=profile.ftype) * Iext
                fixed_inputs.append([obj, Iext, dur])
                continue
            size = np.shape(Iext)[0]
            run_length = dur[1] - dur[0]
            if size != run_length:
                if size == 1:
                    Iext = np.ones(obj.num, dtype=profile.ftype) * Iext
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
