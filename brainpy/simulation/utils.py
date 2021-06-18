# -*- coding: utf-8 -*-

import sys
import time
from collections import Iterable

from brainpy import backend
from brainpy import errors

__all__ = [
  'size2len',
  'check_duration',
  'run_model',
  'format_inputs',
]

SUPPORTED_INPUT_OPS = ['-', '+', 'x', '*', '/', '=']
SUPPORTED_INPUT_TYPE = ['fix', 'iter']


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


def get_run_length_by_duration(duration):
  start, end = check_duration(duration)
  mon_length = int((end - start) / backend.get_dt())
  return mon_length


def run_model(run_func, times, report):
  """Run the model.

  The "run_func" can be the step run function of a dynamical system.

  Parameters
  ----------
  run_func : callable
      The step run function.
  times : iterable
      The model running times.
  report : float
      The percent of the total running length for each report.
  """

  run_length = len(times)
  if report:
    t0 = time.time()
    run_func(_t=times[0], _i=0)
    compile_time = time.time() - t0
    print('Compilation used {:.4f} s.'.format(compile_time))

    print("Start running ...")
    report_gap = int(run_length * report)
    t0 = time.time()
    for run_idx in range(1, run_length):
      run_func(_t=times[run_idx], _i=run_idx)
      if (run_idx + 1) % report_gap == 0:
        percent = (run_idx + 1) / run_length * 100
        print('Run {:.1f}% used {:.3f} s.'.format(percent, time.time() - t0))
    running_time = time.time() - t0
    print('Simulation is done in {:.3f} s.'.format(running_time))
    print()
    return running_time
  else:
    t0 = time.time()
    for run_idx in range(run_length):
      run_func(_t=times[run_idx], _i=run_idx)
    return time.time() - t0


def format_inputs(host, inputs, duration):
  """Format the inputs of a population.

  Parameters
  ----------
  inputs : tuple, list
      The inputs of the population.
  host : DynamicSystem
      The host which contains all data.
  duration : int
      The monitor length.

  Returns
  -------
  formatted_inputs : tuple, list
      The formatted inputs of the population.
  """
  mon_length = get_run_length_by_duration(duration)

  # 1. check inputs
  if inputs is None:
    inputs = []
  if not isinstance(inputs, (tuple, list)):
    raise errors.ModelUseError('"inputs" must be a tuple/list.')
  if len(inputs) > 0 and not isinstance(inputs[0], (list, tuple)):
    if isinstance(inputs[0], str):
      inputs = [inputs]
    else:
      raise errors.ModelUseError('Unknown input structure, only support inputs '
                                 'with format of "(target, value, [type, operation])".')
  for one_input in inputs:
    if not 2 <= len(one_input) <= 4:
      raise errors.ModelUseError('For each target, you must specify '
                                 '"(target, value, [type, operation])".')
    if len(one_input) == 3 and one_input[2] not in SUPPORTED_INPUT_TYPE:
      raise errors.ModelUseError(f'Input type only supports '
                                 f'"{SUPPORTED_INPUT_TYPE}", '
                                 f'not "{one_input[2]}".')
    if len(one_input) == 4 and one_input[3] not in SUPPORTED_INPUT_OPS:
      raise errors.ModelUseError(f'Input operation only supports '
                                 f'"{SUPPORTED_INPUT_OPS}", '
                                 f'not "{one_input[3]}".')

  # 2. format inputs
  # -------------
  nodes = host.nodes()
  formatted_inputs = []
  for one_input in inputs:
    # key
    if not isinstance(one_input[0], str):
      raise errors.ModelUseError('For each input, input[0] must be a string '
                                 'to specify variable of the target.')
    splits = one_input[0].split('.')
    target = '.'.join(splits[:-1])
    key = splits[-1]
    if target == '':
      real_target = host
    else:
      if target not in nodes:
        raise errors.ModelUseError(f'Input target "{target}" is not defined in {host}.')
      real_target = nodes[target]
    if not hasattr(real_target, key):
      raise errors.ModelUseError(f'Input target key "{key}" is not '
                                 f'defined in {real_target}.')

    # value
    value = one_input[1]

    # input type
    if len(one_input) >= 3:
      if one_input[2] == 'iter':
        if not isinstance(value, Iterable):
          raise ValueError(f'Input "{value}" for "{one_input[0]}" is set to '
                           f'be "iter" type, however we got the value with '
                           f'the type of {type(value)}')
        if len(value) < mon_length:
          raise ValueError(f'Input {value} is set to be "iter" type, '
                           f'however it\'s length is less than the duration. '
                           f'This will result in errors in future running.')
      if one_input[2] == 'fix':
        if not isinstance(value, (int, float)):
          raise ValueError(f'Input {value} is set to be "fix" type, '
                           f'however it is a {type(value)}. ')

      data_type = one_input[2]
    else:
      data_type = 'fix'

    # operation
    if len(one_input) == 4:
      operation = one_input[3]
    else:
      operation = '+'

    # final
    format_inp = (one_input[0], value, data_type, operation)
    formatted_inputs.append(format_inp)

  return formatted_inputs
