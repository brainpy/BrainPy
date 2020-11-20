# -*- coding: utf-8 -*-

import typing
from .. import numpy as np
from .. import profile

from ..core_system.neurons import NeuGroup
from ..core_system.neurons import NeuType
from ..core_system.runner import TrajectoryRunner
from ..errors import ModelUseError
from .. import tools


def plot_trajectory(
        neu: NeuType,
        target_vars: typing.Union[typing.List[str], typing.Tuple[str]],
        target_setting: typing.Union[typing.List, typing.Tuple],
        fixed_vars: typing.Dict = None,
        inputs: typing.Union[typing.List, typing.Tuple] = (),
):
    # check initial values
    # ---------------------
    # When target_vars = ['m', 'n']
    # then, target_setting can be: (initial v1, initial v2, duration)
    #                   (0., 1., 100.)       # initial values: m=0., n=1., duration=100.
    #        or,        (0., 1., (10., 90.)) # initial values: m=0., n=1., simulation in [10., 90.]
    #        or,        [(0., 1., (10., 90.)),
    #                    (0.5, 1.5, 100.)]  # two trajectory

    durations = []
    simulating_duration = [np.inf, -np.inf]

    # format target setting
    if isinstance(target_setting[0], (int, float)):
        target_setting = (target_setting,)

    # initial values
    initials = np.zeros((len(target_vars), len(target_setting)), dtype=np.float_)

    # format duration and initial values
    for i, setting in enumerate(target_setting):
        # checking
        try:
            assert isinstance(setting, (tuple, list))
            assert len(setting) == len(target_vars) + 1
        except AssertionError:
            raise ModelUseError('"target_setting" be a tuple with the format of '
                                '(var1 initial, var2 initial, ..., duration).')
        # duration
        duration = setting[-1]
        if isinstance(duration, (int, float)):
            durations.append([0., duration])
            if simulating_duration[0] > 0.:
                simulating_duration[0] = 0.
            if simulating_duration[1] < duration:
                simulating_duration[1] = duration
        elif isinstance(duration, (tuple, list)):
            try:
                assert len(duration) == 2
                assert duration[0] < duration[1]
            except AssertionError:
                raise ModelUseError('duration specification must be a tuple/list with '
                                    'the form of (start, end).')
            durations.append(list(duration))
            if simulating_duration[0] > duration[0]:
                simulating_duration[0] = duration[0]
            if simulating_duration[1] < duration[1]:
                simulating_duration[1] = duration[1]
        else:
            raise ValueError(f'Unknown duration type "{type(duration)}", {duration}')
        # initial values
        for j, val in enumerate(setting[:-1]):
            initials[j, i] = val

    # initialize neuron group
    num = len(target_setting) if len(target_setting) else 1
    group = NeuGroup(neu, geometry=num, monitors=target_vars)
    for i, key in enumerate(target_vars):
        group.ST[key] = initials[i]

    # initialize runner
    group.runner = TrajectoryRunner(group, target_vars=target_vars, fixed_vars=fixed_vars)

    # run
    group.run(duration=simulating_duration, inputs=inputs)

    # monitors
    trajectories = []
    times = group.mon.ts
    dt = profile.get_dt()
    for i, setting in enumerate(target_setting):
        duration = durations[i]
        start = int((duration[0] - simulating_duration[0]) / dt)
        end = int((duration[1] - simulating_duration[0]) / dt)
        trajectory = tools.DictPlus()
        legend = ''
        trajectory['ts'] = times[start: end]
        for j, var in enumerate(target_vars):
            legend += f'${var}_0$={setting[j]}, '
            trajectory[var] = getattr(group.mon, var)[start: end, i]
        if legend.strip():
            trajectory['legend'] = legend[:-2]
        trajectories.append(trajectory)
    return trajectories
