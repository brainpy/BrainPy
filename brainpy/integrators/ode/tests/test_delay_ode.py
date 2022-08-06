# -*- coding: utf-8 -*-

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
  runner = bp.integrators.IntegratorRunner(dde,
                                           args=args,
                                           monitors=monitors,
                                           dt=dt,
                                           inits=inits,
                                           progress_bar=False)
  runner.run(duration)
  return runner.mon




class TestFirstOrderConstantDelay(parameterized.TestCase):
  @staticmethod
  def eq1(x, t, xdelay):
    return -xdelay(t - 1)

  def __init__(self, *args, **kwargs):
    super(TestFirstOrderConstantDelay, self).__init__(*args, **kwargs)

    case1_delay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')
    case2_delay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='linear_interp')
    self.ref1 = delay_odeint(20., self.eq1, args={'xdelay': case1_delay}, state_delays={'x': case1_delay}, method='euler')
    self.ref2 = delay_odeint(20., self.eq1, args={'xdelay': case2_delay}, state_delays={'x': case2_delay}, method='euler')

  @parameterized.named_parameters(
    {'testcase_name': f'constant_delay_{name}',
     'method': name}
    for name in get_supported_methods()
  )
  def test1(self, method):
    case1_delay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='round')
    case2_delay = bm.TimeDelay(bm.zeros((1,)), 1., before_t0=-1., interp_method='linear_interp')

    case1 = delay_odeint(20., self.eq1, args={'xdelay': case1_delay}, state_delays={'x': case1_delay}, method=method)
    case2 = delay_odeint(20., self.eq1, args={'xdelay': case2_delay}, state_delays={'x': case2_delay}, method=method)

    self.assertTrue((case1['x'] - self.ref1['x']).mean() < 1e-3)
    self.assertTrue((case2['x'] - self.ref2['x']).mean() < 1e-3)

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
  @staticmethod
  def eq(x, t, xdelay):
    return -xdelay(t - 2)

  def __init__(self, *args, **kwargs):
    super(TestNonConstantHist, self).__init__(*args, **kwargs)
    delay1 = bm.TimeDelay(bm.zeros(1), 2., before_t0=lambda t: bm.exp(-t) - 1, dt=0.01, interp_method='round')
    self.ref1 = delay_odeint(4., self.eq, args={'xdelay': delay1}, state_delays={'x': delay1}, dt=0.01)
    delay1 = bm.TimeDelay(bm.zeros(1), 2., before_t0=lambda t: bm.exp(-t) - 1, dt=0.01)
    self.ref2 = delay_odeint(4., self.eq, args={'xdelay': delay1}, state_delays={'x': delay1}, dt=0.01)

  @parameterized.named_parameters(
    {'testcase_name': f'constant_delay_{name}', 'method': name}
    for name in get_supported_methods()
  )
  def test1(self, method):
    delay1 = bm.TimeDelay(bm.zeros(1), 2., before_t0=lambda t: bm.exp(-t)-1, dt=0.01, interp_method='round')
    delay2 = bm.TimeDelay(bm.zeros(1), 2., before_t0=lambda t: bm.exp(-t)-1, dt=0.01)
    case1 = delay_odeint(4., self.eq, args={'xdelay': delay1}, state_delays={'x': delay1}, dt=0.01, method=method)
    case2 = delay_odeint(4., self.eq, args={'xdelay': delay2}, state_delays={'x': delay2}, dt=0.01, method=method)

    self.assertTrue((case1['x'] - self.ref1['x']).mean() < 1e-1)
    self.assertTrue((case2['x'] - self.ref2['x']).mean() < 1e-1)


