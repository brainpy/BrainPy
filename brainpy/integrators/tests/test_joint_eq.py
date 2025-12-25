# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import unittest

import brainpy.math as bm
from brainpy._errors import DiffEqError
from brainpy.integrators.joint_eq import _get_args, JointEq


class TestGetArgs(unittest.TestCase):
    def test_POSITIONAL_OR_KEYWORD(self):
        def f(a, b, t, c, d=1.):
            pass

        print(_get_args(f))

    def test_VAR_POSITIONAL(self):
        def f(a, b, t, *c, d=1.):
            pass

        with self.assertRaises(DiffEqError):
            _get_args(f)

    def test_KEYWORD_ONLY(self):
        def f(a, b, t, *, d=1.):
            pass

        with self.assertRaises(DiffEqError):
            _get_args(f)

    # def test_POSITIONAL_ONLY(self):
    #     def f(a, b, t, /, d=1.):
    #       pass
    #
    #     with self.assertRaises(DiffEqError):
    #       _get_args(f)

    def test_VAR_KEYWORD(self):
        def f(a, b, t, **kwargs):
            pass

        with self.assertRaises(DiffEqError):
            _get_args(f)


ENa, gNa = 50., 120.
EK, gK = -77., 36.
EL, gL = -54.387, 0.03
C = 1.0


def dm(m, t, V):
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    return alpha * (1 - m) - beta * m


def dh(h, t, V):
    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    return alpha * (1 - h) - beta * h


def dn(n, t, V):
    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n
    return dndt


def dV(V, t, m, h, n, I):
    I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
    I_K = (gK * n ** 4.0) * (V - EK)
    I_leak = gL * (V - EL)
    dVdt = (- I_Na - I_K - I_leak + I) / C
    return dVdt


class TestJointEqs(unittest.TestCase):
    # def test_variables1(self):
    #   je = JointEq([dV, dn])
    #   with self.assertRaises(DiffEqError):
    #     je(10., 1., 0., I=0.1)

    def test_variables2(self):
        # with self.assertRaises(DiffEqError):
        EQ = JointEq((dV,))
        EQ = JointEq(dV)

    def test_call1(self):
        je1 = JointEq([dV, dn])
        res1 = je1(10., 1., 0., I=0.1, m=0.5, h=0.5)
        je2 = JointEq(dV, dn)
        res2 = je2(10., 1., 0., I=0.1, m=0.5, h=0.5)
        self.assertTrue(res1 == res2)

    def test_do_not_change_par_position(self):
        EQ = JointEq((dV,))
        self.assertEqual(EQ(10., 0., 0.1, 0.2, 0.3, 0.),
                         EQ(V=10., t=0., m=0.1, h=0.2, n=0.3, I=0.))

    def test_return_is_list(self):
        EQ = JointEq((dV,))
        self.assertTrue(isinstance(EQ(V=10., t=0., m=0.1, h=0.2, n=0.3, I=0.), list))
        EQ = JointEq(dV)
        self.assertTrue(isinstance(EQ(V=10., t=0., m=0.1, h=0.2, n=0.3, I=0.), list))

    def test_nested_joint_eq1(self):
        EQ1 = JointEq((dm, dh))
        EQ2 = JointEq((EQ1, dn))
        EQ3 = JointEq((EQ2, dV))
        print(EQ3(m=0.1, h=0.2, n=0.3, V=10., t=0., I=0.))

        EQ1 = JointEq(dm, dh)
        EQ2 = JointEq(EQ1, dn)
        EQ3 = JointEq(EQ2, dV)
        print(EQ3(m=0.1, h=0.2, n=0.3, V=10., t=0., I=0.))

    def test_second_order_ode(self):
        """Test second-order ODE system (e.g., harmonic oscillator)"""
        # Second-order ODE: d²x/dt² = -k*x - c*dx/dt
        # Split into: dx/dt = v, dv/dt = -k*x - c*v
        k = 1.0  # spring constant
        c = 0.1  # damping

        def dx(x, t, v):
            """dx/dt = v"""
            return v

        def dv(v, t, x):
            """dv/dt = -k*x - c*v"""
            return -k * x - c * v

        # Create joint equation
        eq = JointEq(dx, dv)

        # Test call
        result = eq(x=1.0, v=0.0, t=0.0)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 0.0)  # dx/dt = v = 0
        self.assertEqual(result[1], -k * 1.0)  # dv/dt = -k*x

    def test_second_order_ode_wrong_signature(self):
        """Test that wrong signature gives helpful error message"""

        # WRONG: both x and v before t in dx function
        def dx_wrong(x, v, t):
            return v

        def dv(v, t, x):
            return -x

        # This should raise an error with helpful message
        with self.assertRaises(DiffEqError) as cm:
            JointEq(dx_wrong, dv)

        # Check that error message is helpful
        error_msg = str(cm.exception)
        self.assertIn('state variable', error_msg.lower())
        self.assertIn('AFTER "t"', error_msg)
        self.assertIn('dependency', error_msg.lower())
