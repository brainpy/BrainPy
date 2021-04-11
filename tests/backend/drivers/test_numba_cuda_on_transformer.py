# -*- coding: utf-8 -*-

import inspect
import ast
import brainpy as bp
from brainpy.backend.drivers.numba_cuda import _CUDATransformer

bp.backend.set('numba-cuda', dt=0.02)


def test1():
    class LIF(bp.NeuGroup):
        target_backend = 'general'

        def __init__(self, size, t_refractory=1., V_rest=0.,
                     V_reset=-5., V_th=20., R=1., tau=10., **kwargs):
            # parameters
            self.V_rest = V_rest
            self.V_reset = V_reset
            self.V_th = V_th
            self.R = R
            self.tau = tau
            self.t_refractory = t_refractory

            # variables
            self.t_last_spike = bp.ops.ones(size) * -1e7
            self.refractory = bp.ops.zeros(size)
            self.input = bp.ops.zeros(size)
            self.spike = bp.ops.zeros(size)
            self.V = bp.ops.ones(size) * V_reset

            self.int_V = bp.sdeint(f=self.f_v, g=self.g_v)

            super(LIF, self).__init__(size=size, **kwargs)

        @staticmethod
        def f_v(V, t, Iext, V_rest, R, tau):
            return (- (V - V_rest) + R * Iext) / tau

        @staticmethod
        def g_v(V, t, Iext, V_rest, R, tau):
            return 1.

        def update(self, _t):
            for i in range(self.num):
                if _t - self.t_last_spike[i] <= self.t_refractory:
                    self.refractory[i] = 1.
                else:
                    self.refractory[0] = 0.
                    V = self.int_V(self.V[i], _t, self.input[i], self.V_rest, self.R, self.tau)
                    if V >= self.V_th:
                        self.V[i] = self.V_reset
                        self.spike[i] = 1.
                        self.t_last_spike[i] = _t
                    else:
                        self.spike[i] = 0.
                        self.V[i] = V
                self.input[i] = 0.

    lif = LIF(10)

    update_code = bp.tools.deindent(inspect.getsource(lif.update))
    tree = _CUDATransformer(host=lif).visit(ast.parse(update_code))
    tree = ast.fix_missing_locations(tree)
    new_code = bp.tools.ast2code(tree)
    print(new_code)


def test2():
    class LIF(bp.NeuGroup):
        target_backend = 'general'

        def __init__(cls, size, t_refractory=1., V_rest=0.,
                     V_reset=-5., V_th=20., R=1., tau=10., **kwargs):
            # parameters
            cls.V_rest = V_rest
            cls.V_reset = V_reset
            cls.V_th = V_th
            cls.R = R
            cls.tau = tau
            cls.t_refractory = t_refractory

            # variables
            cls.t_last_spike = bp.ops.ones(size) * -1e7
            cls.refractory = bp.ops.zeros(size)
            cls.input = bp.ops.zeros(size)
            cls.spike = bp.ops.zeros(size)
            cls.V = bp.ops.ones(size) * V_reset

            cls.int_V = bp.sdeint(f=cls.f_v, g=cls.g_v)

            super(LIF, cls).__init__(size=size, **kwargs)

        @staticmethod
        def f_v(V, t, Iext, V_rest, R, tau):
            return (- (V - V_rest) + R * Iext) / tau

        @staticmethod
        def g_v(V, t, Iext, V_rest, R, tau):
            return 1.

        def update(cls, _t):
            for i in range(cls.num):
                if _t - cls.t_last_spike[i] <= cls.t_refractory:
                    cls.refractory[i] = 1.
                else:
                    cls.refractory[0] = 0.
                    V = cls.int_V(cls.V[i], _t, cls.input[i], cls.V_rest, cls.R, cls.tau)
                    if V >= cls.V_th:
                        cls.V[i] = cls.V_reset
                        cls.spike[i] = 1.
                        cls.t_last_spike[i] = _t
                    else:
                        cls.spike[i] = 0.
                        cls.V[i] = V
                cls.input[i] = 0.

    lif = LIF(10)

    update_code = bp.tools.deindent(inspect.getsource(lif.update))
    tree = _CUDATransformer(host=lif).visit(ast.parse(update_code))
    # tree = ast.fix_missing_locations(tree)
    new_code = bp.tools.ast2code(tree)
    print(new_code)


test1()
test2()


