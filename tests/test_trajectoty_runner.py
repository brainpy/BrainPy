# -*- coding: utf-8 -*-

from pprint import pprint

import brainpy as bp
import numpy as np
from brainpy.backend.runners.numba_cpu_runner import TrajectoryNumbaRunner
import matplotlib.pyplot as plt

bp.profile.set(jit=True, show_code=True)


if __name__ == '__main__1':
    # HH neuron model
    # ------------------
    code_str = '''
    m = np.clip(int_m(ST['m'], _t_, ST['V']), 0., 1.)
    h = np.clip(int_h(ST['h'], _t_, ST['V']), 0., 1.)
    n = np.clip(int_n(ST['n'], _t_, ST['V']), 0., 1.)
    V = int_V(ST['V'], _t_, m, h, n, ST['inp'])
    sp = np.logical_and(ST['V'] < Vth, V >= Vth)
    ST['sp'] = sp
    ST['V'] = V
    ST['m'] = m
    ST['h'] = h
    ST['n'] = n
    ST['inp'] = 0.
    '''

    formatter = bp.tools.format_code_for_trajectory(code_str, {'m': 0., 'n': 1.})
    pprint(formatter.lines)

if __name__ == '__main__1':
    # HH neuron model
    # ------------------
    code_str = '''
    ST['m'] = np.clip(int_m(ST['m'], _t_, ST['V']), 0., 1.)
    ST['h'] = np.clip(int_h(ST['h'], _t_, ST['V']), 0., 1.)
    ST['n'] = np.clip(int_n(ST['n'], _t_, ST['V']), 0., 1.)
    V = int_V(ST['V'], _t_, m, h, n, ST['inp'])
    sp = np.logical_and(ST['V'] < Vth, V >= Vth)
    ST['sp'] = sp
    ST['V'] = V
    ST['inp'] = 0.
    '''

    formatter = bp.tools.format_code_for_trajectory(code_str, {'m': 0., 'n': 1.})
    pprint(formatter.lines)

if __name__ == '__main__1':
    # Izhikevich
    # --------------
    code_str = '''
            if (_t_ - ST['sp_t']) > ref:
                V = int_V(ST['V'], _t_, ST['u'], ST['inp'])
                u = int_u(ST['u'], _t_, ST['V'])
                if V >= Vth:
                    V = c
                    u += d
                    ST['sp_t'] = _t_
                    ST['sp'] = True
                ST['V'] = V
                ST['u'] = u
                ST['inp'] = 0.
    '''

    formatter = bp.tools.format_code_for_trajectory(code_str, {'V': 0.})
    pprint(formatter.lines)

if __name__ == '__main__1':
    # LIF neuron model
    # ---------------------

    code_str = '''
        if _t_ - ST['sp_t'] > ref:
            V = int_f(ST['V'], _t_, ST['inp'])
            if V >= Vth:
                V = Vr
                ST['sp_t'] = _t_
                ST['sp'] = True
            ST['V'] = V
        else:
            ST['sp'] = False
        ST['inp'] = 0.
    '''
    formatter = bp.tools.format_code_for_trajectory(code_str, {'V': 0.})
    pprint(formatter.lines)



noise = 0.
E_Na = 50.
g_Na = 120.
E_K = -77.
g_K = 36.
E_Leak = -54.387
g_Leak = 0.03
C = 1.0
Vth = 20.

ST = bp.types.NeuState(
    {'V': -65., 'm': 0., 'h': 0., 'n': 0., 'sp': 0., 'inp': 0.},
)


@bp.integrate
def int_m(m, t, V):
    alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta = 4.0 * np.exp(-(V + 65) / 18)
    return alpha * (1 - m) - beta * m


@bp.integrate
def int_h(h, t, V):
    alpha = 0.07 * np.exp(-(V + 65) / 20.)
    beta = 1 / (1 + np.exp(-(V + 35) / 10))
    return alpha * (1 - h) - beta * h


@bp.integrate
def int_n(n, t, V):
    alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    beta = 0.125 * np.exp(-(V + 65) / 80)
    return alpha * (1 - n) - beta * n


@bp.integrate
def int_V(V, t, m, h, n, Isyn):
    INa = g_Na * m ** 3 * h * (V - E_Na)
    IK = g_K * n ** 4 * (V - E_K)
    IL = g_Leak * (V - E_Leak)
    dvdt = (- INa - IK - IL + Isyn) / C
    return dvdt, noise / C


def update(ST, _t_):
    m = np.clip(int_m(ST['m'], _t_, ST['V']), 0., 1.)
    h = np.clip(int_h(ST['h'], _t_, ST['V']), 0., 1.)
    n = np.clip(int_n(ST['n'], _t_, ST['V']), 0., 1.)
    V = int_V(ST['V'], _t_, m, h, n, ST['inp'])
    sp = np.logical_and(ST['V'] < Vth, V >= Vth)
    ST['sp'] = sp
    ST['V'] = V
    ST['m'] = m
    ST['h'] = h
    ST['n'] = n
    ST['inp'] = 0.


HH = bp.NeuType(name='HH_neuron', requires={"ST": ST}, steps=update, vector_based=True)

tau = 10.
Vr = 0.
Vth2 = 10.
ref = 0.

ST = bp.types.NeuState(
    {'V': 0, 'sp_t': -1e7, 'sp': 0., 'inp': 0.},
)


@bp.integrate
def int_f(V, t, Isyn):
    return (-V + Vr + Isyn) / tau


def update(ST, _t_):
    if _t_ - ST['sp_t'] > ref:
        V = int_f(ST['V'], _t_, ST['inp'])
        if V >= Vth2:
            V = Vr
            ST['sp_t'] = _t_
            ST['sp'] = True
        ST['V'] = V
    else:
        ST['sp'] = False
    ST['inp'] = 0.


LIF = bp.NeuType(name='LIF', requires=dict(ST=ST), steps=update, vector_based=False)


a=0.02
b=0.20
c=-65.
d=8.
Vth3=30.
mode=None

state = bp.types.NeuState(
    {'V': 0., 'u': 1., 'sp': 0., 'sp_t': -1e7, 'inp': 0.},
)


@bp.integrate
def int_u3(u, t, V):
    return a * (b * V - u)

@bp.integrate
def int_V3(V, t, u, Isyn):
    dfdt = 0.04 * V * V + 5 * V + 140 - u + Isyn
    return dfdt


def update(ST, _t_):
    ST['V'] = int_V3(ST['V'], _t_, ST['u'], ST['inp'])
    ST['u'] = int_u3(ST['u'], _t_, ST['V'])
    if ST['V'] >= Vth3:
        ST['V'] = c
        ST['u'] += d
        ST['sp_t'] = _t_
        ST['sp'] = True
    ST['inp'] = 0.

Izhikevich = bp.NeuType(name='Izhikevich', requires={'ST': state}, steps=update, vector_based=False)


if __name__ == '__main__1':
    group = bp.NeuGroup(HH, size=10)

    runner = TrajectoryNumbaRunner(group, target_vars=['m', 'h'])
    print(runner.target_vars)
    print(runner.fixed_vars)


def get_trajectories(
        model: NeuType,
        target_vars: typing.Union[typing.List[str], typing.Tuple[str]],
        initials: typing.Union[typing.List, typing.Tuple],
        duration: typing.Union[int, typing.List, typing.Tuple],
        fixed_vars: typing.Dict = None,
        inputs: typing.Union[typing.List, typing.Tuple] = (),
        pars_update: typing.Dict = None,
):
    """Get trajectories.

    Parameters
    ----------
    model : NeuType
        The neuron model.
    target_vars : list, tuple
        The target variables.
    initials : tuple, list
        The initial value setting of the targets. It can be a data with the format of
        ``[(v1, v2), (v1, v2)]``.
    duration : int, float, tuple, list
        The running duration. Same with the ``duration`` in ``Network.run()``.
    inputs : list, tuple
        The model inputs. Same with the ``inputs`` in ``Network.run()``.
    fixed_vars : dict
        The fixed variables.
    pars_update : dict
        The parameters to update.

    Returns
    -------
    trajectories : list
        The trajectories.
    """

    # format initial value setting
    if isinstance(initials[0], (int, float)):
        initials = [initials, ]
    initials = np.array(initials)

    # format duration and initial values
    initial_states = np.zeros((len(target_vars), len(initials)), dtype=np.float_)
    for i, initial in enumerate(initials):
        # checking
        try:
            assert len(initial) == len(target_vars)
        except AssertionError:
            raise ModelUseError('"initials" be a tuple/list/array with the format of '
                                '[[var1 initial, var2 initial]].')
        # initial values
        for j, val in enumerate(initial):
            initial_states[j, i] = val

    # initialize neuron group
    num = len(initials)
    group = NeuGroup(model, geometry=num, monitors=target_vars, pars_update=pars_update)
    for j, key in enumerate(target_vars):
        group.ST[key] = initial_states[j]
    group.runner = TrajectoryNumbaRunner(group, target_vars=target_vars, fixed_vars=fixed_vars)
    group.run(duration=duration, inputs=inputs)

    # monitors
    trajectories = []
    for i, initial in enumerate(initials):
        trajectory = tools.DictPlus()
        legend = 'traj, '
        trajectory['ts'] = group.mon.ts
        for j, var in enumerate(target_vars):
            legend += f'${var}_0$={initial[j]}, '
            trajectory[var] = getattr(group.mon, var)[:, i]
        if legend.strip():
            trajectory['legend'] = legend[:-2]
        trajectories.append(trajectory)
    return trajectories

if __name__ == '__main__':
    pass
    from brainpy.analysis.phase_plane import get_trajectories

    # Try LIF neuron model
    trajectories = get_trajectories(LIF, target_vars=['V'], initials=(0., (20., 100.)), inputs=('ST.inp', 12.))
    plt.plot(trajectories[0].ts, trajectories[0].V, label=trajectories[0].legend)
    plt.legend()
    plt.show()

    # Try LIF neuron model: variable "V"
    trajectories = get_trajectories(Izhikevich, target_vars=['V'], initials=(0., (20., 100.)),
                                    inputs=('ST.inp', 12.))
    plt.plot(trajectories[0].ts, trajectories[0].V, label=trajectories[0].legend)
    plt.legend()
    plt.show()

    # Try LIF neuron model: variable "u"
    trajectories = get_trajectories(Izhikevich, target_vars=['u'], initials=(0., (20., 100.)),
                                    inputs=('ST.inp', 12.))
    plt.plot(trajectories[0].ts, trajectories[0].u, label=trajectories[0].legend)
    plt.legend()
    plt.show()

    # Try HH neuron model: variable 'V', 'n'
    trajectories = get_trajectories(HH,
                                    target_vars=['V', 'n'],
                                    initials=[(-65., 0., (20., 100.)),
                                              (-75., 0.1, 60.)],
                                    inputs=('ST.inp', 3.))
    plt.plot(trajectories[0].ts, trajectories[0].V, label=trajectories[0].legend)
    plt.plot(trajectories[1].ts, trajectories[1].V, label=trajectories[1].legend)
    plt.legend(title='Initial values')
    plt.show()


