# -*- coding: utf-8 -*-

import ast
import inspect
from pprint import pprint

import brainpy as bp
from brainpy.backend.drivers.numba_cuda import _CudaStepFuncReader


class LIF(bp.NeuGroup):
    target_backend = ['numba', 'numpy']

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

        super(LIF, self).__init__(size=size, **kwargs)

    @staticmethod
    @bp.odeint
    def int_V(V, t, Iext, V_rest, R, tau):
        return (- (V - V_rest) + R * Iext) / tau

    def update(self, _t):
        for i in range(self.size[0]):
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


def test_automic_op(model):
    synapse = model(pre=LIF(1), post=LIF(2))

    update_code = bp.tools.deindent(inspect.getsource(synapse.update))
    formatter = _CudaStepFuncReader(host=synapse)
    formatter.visit(ast.parse(update_code))

    print('lefts:')
    pprint(formatter.lefts)
    print()
    print('rights:')
    pprint(formatter.rights)
    print()
    print('lines:')
    pprint(formatter.lines)
    print()
    print('delay_call:')
    pprint(formatter.delay_call.keys())
    for v in formatter.delay_call.values():
        pprint(v)
    print()


def test_automic_op_in_assign1():
    class Syn(bp.TwoEndConn):
        target_backend = 'numpy'

        def __init__(self, pre, post): super(Syn, self).__init__(pre, post)

        def update(self):
            for i in range(self.post.num):
                self.post.V[i] = self.post.V[i] + 10.

    test_automic_op(Syn)


def test_automic_op_in_assign2():
    class Syn(bp.TwoEndConn):
        target_backend = 'numpy'

        def __init__(self, pre, post): super(Syn, self).__init__(pre, post)

        def update(self):
            for i in range(self.post.num):
                self.post.V[i] = -self.post.V[i] + 10.

    test_automic_op(Syn)


def test_automic_op_in_assign3():
    class Syn(bp.TwoEndConn):
        target_backend = 'numpy'

        def __init__(self, pre, post): super(Syn, self).__init__(pre, post)

        def update(self):
            for i in range(self.post.num):
                self.post.V[i] = + 10. - self.post.V[i]

    test_automic_op(Syn)


def test_automic_op_in_assign4():
    class Syn(bp.TwoEndConn):
        target_backend = 'numpy'

        def __init__(self, pre, post): super(Syn, self).__init__(pre, post)

        def update(self):
            for i in range(self.post.num):
                self.post.V[i] = + 10. + self.post.V[i]

    test_automic_op(Syn)


def test_automic_op_in_assign5():
    class Syn(bp.TwoEndConn):
        target_backend = 'numpy'

        def __init__(self, pre, post): super(Syn, self).__init__(pre, post)

        def update(self):
            for i in range(self.post.num):
                self.post.V[i] = + 10. + (-self.post.V[i])

    test_automic_op(Syn)


def test_automic_op_in_assign6():
    class Syn(bp.TwoEndConn):
        target_backend = 'numpy'

        def __init__(self, pre, post): super(Syn, self).__init__(pre, post)

        def update(self):
            for i in range(self.post.num):
                self.post.V[i] = + 10. + (-2 * self.post.V[i])

    test_automic_op(Syn)


def test_automic_op_in_augassign1():
    class Syn(bp.TwoEndConn):
        target_backend = 'numpy'

        def __init__(self, pre, post): super(Syn, self).__init__(pre, post)

        def update(self):
            for i in range(self.post.num):
                self.post.V[i] += 10

    test_automic_op(Syn)


def test_automic_op_in_augassign2():
    class Syn(bp.TwoEndConn):
        target_backend = 'numpy'

        def __init__(self, pre, post): super(Syn, self).__init__(pre, post)

        def update(self):
            for i in range(self.post.num):
                self.post.V[i] += 10

    test_automic_op(Syn)


def test_automic_op_in_augassign3():
    class Syn(bp.TwoEndConn):
        target_backend = 'numpy'

        def __init__(self, pre, post): super(Syn, self).__init__(pre, post)

        def update(self):
            for i in range(self.post.num):
                self.post.V[i] /= 10

    test_automic_op(Syn)


# test_automic_op_in_assign1()
# test_automic_op_in_assign2()
# test_automic_op_in_assign3()
# test_automic_op_in_assign4()
# test_automic_op_in_assign5()
# test_automic_op_in_assign6()

# test_automic_op_in_augassign1()
# test_automic_op_in_augassign2()
test_automic_op_in_augassign3()
