# -*- coding: utf-8 -*-

import time

from brainpy import backend
from brainpy import errors
from brainpy.backend import ops
from brainpy.simulation import constants

__all__ = [
    'size2len',
    'check_duration',
    'run_model',
    'format_pop_level_inputs',
    'format_net_level_inputs',
]


def size2len(size):
    if isinstance(size, int):
        return size
    elif isinstance(size, (tuple, list)):
        a = 1
        for b in size:
            a *= b
        return a
    else:
        raise ValueError


def check_duration(duration):
    """Check the running duration.

    Parameters
    ----------
    duration : int, list, tuple
        The running duration, it can be an int (which represents the end
        of the simulation), of a tuple/list of int (which represents the
        [start, end] / [end, start] of the simulation).

    Returns
    -------
    duration : tuple
        The tuple of running duration includes (start, end).
    """
    if isinstance(duration, (int, float)):
        start, end = 0., duration
    elif isinstance(duration, (tuple, list)):
        assert len(duration) == 2, 'Only support duration setting with the ' \
                                   'format of "(start, end)" or "end".'
        start, end = duration
    else:
        raise ValueError(f'Unknown duration type: {type(duration)}. Currently, BrainPy only '
                         f'support duration specification with the format of "(start, end)" '
                         f'or "end".')

    if start > end:
        start, end = end, start
    return start, end


def run_model(run_func, times, report, report_percent):
    """Run the model.

    The "run_func" can be the step run function of a population, or a network.

    Parameters
    ----------
    run_func : callable
        The step run function.
    times : iterable
        The model running times.
    report : bool
        Whether report the progress of the running.
    report_percent : float
        The percent of the total running length for each report.
    """
    run_length = len(times)
    dt = backend.get_dt()
    if report:
        t0 = time.time()
        for i, t in enumerate(times[:1]):
            run_func(_t=t, _i=i, _dt=dt)
        print('Compilation used {:.4f} s.'.format(time.time() - t0))

        print("Start running ...")
        report_gap = int(run_length * report_percent)
        t0 = time.time()
        for run_idx in range(1, run_length):
            run_func(_t=times[run_idx], _i=run_idx, _dt=dt)
            if (run_idx + 1) % report_gap == 0:
                percent = (run_idx + 1) / run_length * 100
                print('Run {:.1f}% used {:.3f} s.'.format(percent, time.time() - t0))
        print('Simulation is done in {:.3f} s.'.format(time.time() - t0))
        print()
    else:
        for run_idx in range(run_length):
            run_func(_t=times[run_idx], _i=run_idx, _dt=dt)


def format_pop_level_inputs(inputs, host, mon_length):
    """Format the inputs of a population.

    Parameters
    ----------
    inputs : tuple, list
        The inputs of the population.
    host : Population
        The host which contains all data.
    mon_length : int
        The monitor length.

    Returns
    -------
    formatted_inputs : tuple, list
        The formatted inputs of the population.
    """
    if inputs is None:
        inputs = []
    if not isinstance(inputs, (tuple, list)):
        raise errors.ModelUseError('"inputs" must be a tuple/list.')
    if len(inputs) > 0 and not isinstance(inputs[0], (list, tuple)):
        if isinstance(inputs[0], str):
            inputs = [inputs]
        else:
            raise errors.ModelUseError('Unknown input structure, only support inputs '
                                       'with format of "(key, value, [operation])".')
    for input in inputs:
        if not 2 <= len(input) <= 3:
            raise errors.ModelUseError('For each target, you must specify "(key, value, [operation])".')
        if len(input) == 3 and input[2] not in constants.SUPPORTED_INPUT_OPS:
            raise errors.ModelUseError(f'Input operation only supports '
                                       f'"{list(constants.SUPPORTED_INPUT_OPS.keys())}", '
                                       f'not "{input[2]}".')

    # format inputs
    # -------------
    formatted_inputs = []
    for input in inputs:
        # key
        if not isinstance(input[0], str):
            raise errors.ModelUseError('For each input, input[0] must be a string '
                                       'to specify variable of the target.')
        key = input[0]
        if not hasattr(host, key):
            raise errors.ModelUseError(f'Input target key "{key}" is not defined in {host}.')

        # value and data type
        val = input[1]
        if isinstance(input[1], (int, float)):
            data_type = 'fix'
        else:
            shape = ops.shape(input[1])
            if shape[0] == mon_length:
                data_type = 'iter'
            else:
                data_type = 'fix'

        # operation
        if len(input) == 3:
            operation = input[2]
        else:
            operation = '+'
        if operation not in constants.SUPPORTED_INPUT_OPS:
            raise errors.ModelUseError(f'Currently, BrainPy only support operations '
                                       f'{list(constants.SUPPORTED_INPUT_OPS.keys())}, '
                                       f'not {operation}')
        # input
        format_inp = (key, val, operation, data_type)
        formatted_inputs.append(format_inp)

    return formatted_inputs


def format_net_level_inputs(inputs, run_length):
    """Format the inputs of a network.

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
    from brainpy.simulation.dynamic_system import DynamicSystem

    # 1. format the inputs to standard
    #    formats and check the inputs
    if not isinstance(inputs, (tuple, list)):
        raise errors.ModelUseError('"inputs" must be a tuple/list.')
    if len(inputs) > 0 and not isinstance(inputs[0], (list, tuple)):
        if isinstance(inputs[0], DynamicSystem):
            inputs = [inputs]
        else:
            raise errors.ModelUseError('Unknown input structure. Only supports '
                                       '"(target, key, value, [operation])".')
    for input in inputs:
        if not 3 <= len(input) <= 4:
            raise errors.ModelUseError('For each target, you must specify '
                                       '"(target, key, value, [operation])".')
        if len(input) == 4:
            if input[3] not in constants.SUPPORTED_INPUT_OPS:
                raise errors.ModelUseError(f'Input operation only supports '
                                           f'"{list(constants.SUPPORTED_INPUT_OPS.keys())}", '
                                           f'not "{input[3]}".')

    # 2. format inputs
    formatted_inputs = {}
    for input in inputs:
        # target
        if isinstance(input[0], DynamicSystem):
            target = input[0]
            target_name = input[0].name
        else:
            raise KeyError(f'Unknown input target: {str(input[0])}')

        # key
        key = input[1]
        if not isinstance(key, str):
            raise errors.ModelUseError('For each input, input[1] must be a string '
                                       'to specify variable of the target.')
        if not hasattr(target, key):
            raise errors.ModelUseError(f'Target {target} does not have key {key}. '
                                       f'So, it can not assign input to it.')

        # value and data type
        val = input[2]
        if isinstance(input[2], (int, float)):
            data_type = 'fix'
        else:
            shape = ops.shape(val)
            if shape[0] == run_length:
                data_type = 'iter'
            else:
                data_type = 'fix'

        # operation
        if len(input) == 4:
            operation = input[3]
        else:
            operation = '+'

        # final result
        if target_name not in formatted_inputs:
            formatted_inputs[target_name] = []
        format_inp = (key, val, operation, data_type)
        formatted_inputs[target_name].append(format_inp)
    return formatted_inputs

