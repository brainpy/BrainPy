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
import jax.numpy as jnp
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.integrators.ode import get_supported_methods

block = True


def delay_odeint(duration, eq, args=None, inits=None,
                 state_delays=None, neutral_delays=None,
                 monitors=('x',), method='euler', dt=0.1):
    # define integrators of ODEs based on `brainpy.odeint`
    dde = bp.odeint(eq,
                    state_delays=state_delays,
                    neutral_delays=neutral_delays,
                    method=method)
    # define IntegratorRunner
    runner = bp.IntegratorRunner(dde,
                                 args=args,
                                 monitors=monitors,
                                 dt=dt,
                                 inits=inits,
                                 progress_bar=False)
    runner.run(duration)
    return runner.mon


def get_eq1(xdelay):
    def eq1(x, t):
        return -xdelay(t - 1)

    return eq1


case1_delay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')
case2_delay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='linear_interp')
ref1 = delay_odeint(20., get_eq1(case1_delay), state_delays={'x': case1_delay}, method='euler')
ref2 = delay_odeint(20., get_eq1(case2_delay), state_delays={'x': case2_delay}, method='euler')


def get_eq2(xdelay):
    def eq2(x, t):
        return -xdelay(t - 2)

    return eq2


delay1 = bm.TimeDelay(bm.zeros(1), 2., before_t0=lambda t: jnp.exp(-t) - 1, dt=0.01, interp_method='round')
ref3 = delay_odeint(4., get_eq2(delay1), state_delays={'x': delay1}, dt=0.01)
delay1 = bm.TimeDelay(bm.zeros(1), 2., before_t0=lambda t: jnp.exp(-t) - 1, dt=0.01)
ref4 = delay_odeint(4., get_eq2(delay1), state_delays={'x': delay1}, dt=0.01)


class TestFirstOrderConstantDelay(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFirstOrderConstantDelay, self).__init__(*args, **kwargs)

    @parameterized.named_parameters(
        {'testcase_name': f'constant_delay_{name}',
         'method': name}
        for name in get_supported_methods()
    )
    def test1(self, method):
        bm.random.seed()
        case1_delay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')
        case2_delay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='linear_interp')

        case1 = delay_odeint(20., get_eq1(case1_delay), state_delays={'x': case1_delay}, method=method)
        case2 = delay_odeint(20., get_eq1(case2_delay), state_delays={'x': case2_delay}, method=method)

        print(method)
        print("case1.keys()", case1.keys())
        print("case2.keys()", case2.keys())
        print("self.ref1.keys()", ref1.keys())
        print("self.ref2.keys()", ref2.keys())

        # self.assertTrue((case1['x'] - self.ref1['x']).mean() < 1e-3)
        # self.assertTrue((case2['x'] - self.ref2['x']).mean() < 1e-3)

        # fig, axs = plt.subplots(2, 1)
        # fig.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
        # fig.suptitle("$y'(t)=-y(t-1)$")
        # axs[0].plot(case1.ts, case1.x, color='red', linewidth=1)
        # axs[0].set_title('$ihf(t)=-1$')
        # axs[1].plot(case2.ts, case2.x, color='red', linewidth=1)
        # axs[1].set_title('$ihf(t)=0$')
        # plt.show(block=block)
        # plt.close()


class TestNonConstantHist(parameterized.TestCase):
    def get_eq(self, xdelay):
        def eq(x, t):
            return -xdelay(t - 2)

        return eq

    def __init__(self, *args, **kwargs):
        super(TestNonConstantHist, self).__init__(*args, **kwargs)

    @parameterized.named_parameters(
        {'testcase_name': f'constant_delay_{name}', 'method': name}
        for name in get_supported_methods()
    )
    def test1(self, method):
        bm.random.seed()

        delay1 = bm.TimeDelay(bm.zeros(1), 2., before_t0=lambda t: jnp.exp(-t) - 1, dt=0.01, interp_method='round')
        delay2 = bm.TimeDelay(bm.zeros(1), 2., before_t0=lambda t: jnp.exp(-t) - 1, dt=0.01)
        case1 = delay_odeint(4., self.get_eq(delay1), state_delays={'x': delay1}, dt=0.01, method=method)
        case2 = delay_odeint(4., self.get_eq(delay2), state_delays={'x': delay2}, dt=0.01, method=method)

        print("case1.keys()", case1.keys())
        print("case2.keys()", case2.keys())
        print("ref3.keys()", ref3.keys())
        print("ref4.keys()", ref4.keys())

        # self.assertTrue((case1['x'] - self.ref1['x']).mean() < 1e-1)
        # self.assertTrue((case2['x'] - self.ref2['x']).mean() < 1e-1)
