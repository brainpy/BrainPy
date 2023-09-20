import brainpy.math as bm
import brainpy as bp
from jax import ShapedArray


def abs_eval(events, indices, indptr, *, weight, post_num):
  return [ShapedArray((post_num,), bm.float32), ]


def con_compute(outs, ins):
  post_val, = outs
  post_val.fill(0)
  events, indices, indptr, weight, _ = ins
  weight = weight[()]
  for i in range(events.size):
    if events[i]:
      for j in range(indptr[i], indptr[i + 1]):
        index = indices[j]
        post_val[index] += weight


event_sum = bm.XLACustomOp(eval_shape=abs_eval, cpu_func=con_compute)


class ExponentialV2(bp.synapses.TwoEndConn):
  """Exponential synapse model using customized operator written in C++."""

  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.):
    super(ExponentialV2, self).__init__(pre=pre, post=post, conn=conn)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.E = E
    self.tau = tau
    self.delay = delay
    self.g_max = g_max
    self.pre2post = self.conn.require('pre2post')

    # variables
    self.g = bm.Variable(bm.zeros(self.post.num))

    # function
    self.integral = bp.odeint(lambda g, t: -g / self.tau, method='exp_auto')

  def update(self):
    self.g.value = self.integral(self.g, bp.share['t'])
    self.g += event_sum(self.pre.spike,
                        self.pre2post[0],
                        self.pre2post[1],
                        weight=self.g_max,
                        post_num=self.post.num)[0]
    self.post.input += self.g * (self.E - self.post.V)


class EINet(bp.DynSysGroup):
  def __init__(self, scale):
    super().__init__()
    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.), method='exp_auto')
    self.E = bp.neurons.LIF(int(3200 * scale), **pars)
    self.I = bp.neurons.LIF(int(800 * scale), **pars)

    # synapses
    self.E2E = ExponentialV2(self.E, self.E, bp.conn.FixedProb(prob=0.02), E=0., g_max=0.6 / scale, tau=5.)
    self.E2I = ExponentialV2(self.E, self.I, bp.conn.FixedProb(prob=0.02), E=0., g_max=0.6 / scale, tau=5.)
    self.I2E = ExponentialV2(self.I, self.E, bp.conn.FixedProb(prob=0.02), E=-80., g_max=6.7 / scale, tau=10.)
    self.I2I = ExponentialV2(self.I, self.I, bp.conn.FixedProb(prob=0.02), E=-80., g_max=6.7 / scale, tau=10.)


def test1():
  bm.set_platform('cpu')
  net2 = EINet(scale=0.1)
  runner = bp.DSRunner(net2, inputs=[('E.input', 20.), ('I.input', 20.)])
  r = runner.predict(100., eval_time=True)
  bm.clear_buffer_memory()
