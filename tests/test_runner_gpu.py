# -*- coding: utf-8 -*-

from pprint import pprint

import numpy as np
from numba import cuda

import brainpy as bp
from brainpy.core_system.runner import Runner


def define_lif():
    tau = 10.
    Vr = 0.
    Vth = 10.
    noise = 0.
    ref = 0.

    ST = bp.types.NeuState(
        {'V': 0, 'sp_t': -1e7, 'spike': 0., 'input': 0.},
    )

    @bp.integrate
    def int_f(V, t, Isyn):
        return (-V + Vr + Isyn) / tau, noise / tau

    def update(ST, _t):
        if _t - ST['sp_t'] > ref:
            V = int_f(ST['V'], _t, ST['input'])
            if V >= Vth:
                V = Vr
                ST['sp_t'] = _t
                ST['spike'] = True
            ST['V'] = V
        else:
            ST['spike'] = False
        ST['input'] = 0.

    return bp.NeuType(name='LIF',
                      ST=ST,
                      steps=update,
                      mode='scalar')


def test_input_fix():
    if not cuda.is_available():
        return

    bp.profile.set(jit=True, device='gpu')

    lif = define_lif()

    num = 100
    group = bp.NeuGroup(lif, geometry=(num,))

    runner = Runner(group)
    res = runner.get_codes_of_input([('ST.input', 1., '=', 'fix')])
    assert res['input-0']['num_data'] == num
    assert res['input-0']['codes'][-1].endswith('ST_input_inp')
    pprint(res)

    print('\n' * 3)

    runner = Runner(group)
    res = runner.get_codes_of_input([('ST.input', np.random.random(100), '=', 'fix')])
    assert res['input-0']['num_data'] == num
    assert res['input-0']['codes'][-1].endswith('ST_input_inp[cuda_i]')

    pprint(res)


def test_input_iter():
    if not cuda.is_available():
        return

    bp.profile.set(jit=True, device='gpu')
    lif = define_lif()
    num = 100
    group = bp.NeuGroup(lif, geometry=(num,))

    runner = Runner(group)
    res = runner.get_codes_of_input([('ST.input', np.random.random(1000), '=', 'iter')])
    assert res['input-0']['num_data'] == num
    assert res['input-0']['codes'][-1].endswith('ST_input_inp[_i]')
    pprint(res)

    print('\n' * 3)

    runner = Runner(group)
    res = runner.get_codes_of_input([('ST.input', np.random.random((1000, num)), '=', 'iter')])
    assert res['input-0']['num_data'] == num
    assert res['input-0']['codes'][-1].endswith('ST_input_inp[_i, cuda_i]')
    pprint(res)


def test_monitor():
    if not cuda.is_available():
        return

    bp.profile.set(jit=True, device='gpu')

    lif = define_lif()

    num = 100
    group = bp.NeuGroup(lif, geometry=(num,), monitors=['spike'])

    runner = Runner(group)
    mon, res = runner.get_codes_of_monitor([('ST.spike', None)], 1000)
    pprint(res)
    assert res['monitor-0']['num_data'] == num
    assert res['monitor-0']['codes'][-1].endswith('ST[2, cuda_i]')

    pprint(res)
    print('\n' * 4)

    runner = Runner(group)
    mon, res = runner.get_codes_of_monitor([('ST.spike', [1, 2, 4])], 1000)
    assert res['monitor-0']['num_data'] == 3
    assert res['monitor-0']['codes'][-1].endswith('= ST[2, mon_idx]')
    pprint(res)
