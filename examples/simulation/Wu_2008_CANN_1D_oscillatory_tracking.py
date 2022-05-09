# Implementation of the paper:
#
# - Si Wu, Kosuke Hamaguchi, and Shun-ichi Amari. "Dynamics and computation
#   of continuous attractors." Neural computation 20.4 (2008): 994-1025.

import brainpy as bp
import brainpy.math as bm


class CANN1D(bp.dyn.NeuGroup):
  def __init__(self, num, tau=1., tau_v=50., k=1., a=0.3, A=0.2, J0=1.,
               z_min=-bm.pi, z_max=bm.pi, m=0.3):
    super(CANN1D, self).__init__(size=num)

    # parameters
    self.tau = tau  # The synaptic time constant
    self.tau_v = tau_v
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value
    self.m = m

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, num)  # The encoded feature values
    self.rho = num / self.z_range  # The neural density
    self.dx = self.z_range / num  # The stimulus density

    # The connection matrix
    self.conn_mat = self.make_conn()

    # variables
    self.r = bm.Variable(bm.zeros(num))
    self.u = bm.Variable(bm.zeros(num))
    self.v = bm.Variable(bm.zeros(num))
    self.input = bm.Variable(bm.zeros(num))

  def dist(self, d):
    d = bm.remainder(d, self.z_range)
    d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
    return d

  def make_conn(self):
    x_left = bm.reshape(self.x, (-1, 1))
    x_right = bm.repeat(self.x.reshape((1, -1)), len(self.x), axis=0)
    d = self.dist(x_left - x_right)
    conn = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
    return conn

  def get_stimulus_by_pos(self, pos):
    return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

  def update(self, t, dt):
    r1 = bm.square(self.u)
    r2 = 1.0 + self.k * bm.sum(r1)
    self.r.value = r1 / r2
    Irec = bm.dot(self.conn_mat, self.r)
    self.u.value = self.u + (-self.u + Irec + self.input - self.v) / self.tau * dt
    self.v.value = self.v + (-self.v + self.m * self.u) / self.tau_v * dt
    self.input[:] = 0.


cann = CANN1D(num=512)

# Smooth tracking #
dur1, dur2, dur3 = 100., 2000., 500.
num1 = int(dur1 / bm.get_dt())
num2 = int(dur2 / bm.get_dt())
num3 = int(dur3 / bm.get_dt())
position = bm.zeros(num1 + num2 + num3)
final_pos = cann.a / cann.tau_v * 0.6 * dur2
position[num1: num1 + num2] = bm.linspace(0., final_pos, num2)
position[num1 + num2:] = final_pos
position = position.reshape((-1, 1))
Iext = cann.get_stimulus_by_pos(position)
runner = bp.dyn.DSRunner(cann,
                         inputs=('input', Iext, 'iter'),
                         monitors=['u', 'v'],
                         dyn_vars=cann.vars())
runner(dur1 + dur2 + dur3)
bp.visualize.animate_1D(
  dynamical_vars=[
    {'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
    {'ys': runner.mon.v, 'xs': cann.x, 'legend': 'v'},
    {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}
  ],
  frame_step=30,
  frame_delay=5,
  show=True,
  save_path='./cann_1d_oscillatory_tracking.gif'
)
