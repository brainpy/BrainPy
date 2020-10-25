from brainpy.core_system import types
from brainpy.core_system.neurons import NeuGroup, NeuType
import brainpy.numpy as np
from brainpy import profile

_dt = 0.02


def main(E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387, g_Leak=0.03, C=1.0, Vth=20.):
    attrs = dict(
        ST=types.NeuState(['V', 'm', 'h', 'n', 'sp', 'inp']),
        pre=types.NeuState(['V', 'm', 'h', 'n', 'sp', 'inp']),
        x=types.Array(dim=1),
    )

    def step(ST):
        V = ST['V']
        m = ST['m']
        h = ST['h']
        n = ST['n']
        input = ST['inp']

        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = m + dmdt * _dt

        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = h + dhdt * _dt

        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = n + dndt * _dt

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + input) / C
        V += dvdt * _dt

        ST['V'] = V
        ST['m'] = m
        ST['h'] = h
        ST['n'] = n
        ST['sp'] = V > Vth
        ST['inp'] = 10.

    return {'attrs': attrs, 'step_func': step}


HH = NeuType('HH', main, vector_based=True)



def try_add_input():
    group = NeuGroup(HH, (10, 10), )
    group.pre = types.NeuState(['V', 'm', 'h', 'n', 'sp', 'inp'])(100)

    key_val_types = [
        ('inp', 1., '=', 'fix'),
        ('ST.V', np.zeros(10), '=', 'iter'),
        ('ST.inp', np.zeros(10), '+', 'iter'),
        ('pre.inp', np.zeros(10), '/', 'iter'),
    ]

    # profile._backend = 'numpy'
    # group._add_input(key_val_types)

    profile._backend = 'numba'
    group._add_input(key_val_types)



def try_add_monitor():
    # Case 1
    print('-'*20)
    print('Case 1: without index')
    print('-'*20)
    group = NeuGroup(HH, (10, 10), monitors=['V', 'm', 'pre.h', 'x'])
    group.pre = types.NeuState(['V', 'm', 'h', 'n', 'sp', 'inp'])(100)
    group.x = np.zeros(100)

    profile._backend = 'numpy'
    group._add_monitor(1000)
    print('-' * 30)
    profile._backend = 'numba'
    group._add_monitor(1000)

    # Case 2
    profile._backend = 'numpy'
    print('-' * 20)
    print('Case 2: with index')
    print('-' * 20)
    group = NeuGroup(HH, (10, 10), monitors=[('V', [1,2]), 'm', ('pre.h', np.arange(10)), 'x'])
    group.pre = types.NeuState(['V', 'm', 'h', 'n', 'sp', 'inp'])(100)
    group.x = np.zeros(100)

    group._add_monitor(1000)
    print('-' * 30)
    profile._backend = 'numba'
    group._add_monitor(1000)




try_add_monitor()

