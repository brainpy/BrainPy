# -*- coding: utf-8 -*-

import sys

sys.path.append(r'/mnt/d/codes/Projects/BrainPy')

import brainpy as bp

bp.math.use_backend('jax')


class CANN1D(bp.NeuGroup):
  def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4.,
               z_min=-bp.math.pi, z_max=bp.math.pi, **kwargs):
    # parameters
    self.tau = tau  # The synaptic time constant
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bp.math.linspace(z_min, z_max, num)  # The encoded feature values

    # variables
    self.u = bp.math.zeros(num)
    self.input = bp.math.zeros(num)

    # The connection matrix
    self.conn_mat = self.make_conn(self.x)

    super(CANN1D, self).__init__(size=num, **kwargs)

    self.rho = num / self.z_range  # The neural density
    self.dx = self.z_range / num  # The stimulus density

  def dist(self, d):
    d = bp.math.remainder(d, self.z_range)
    d = bp.math.where(d > 0.5 * self.z_range, d - self.z_range, d)
    return d

  def make_conn(self, x):
    assert bp.math.ndim(x) == 1
    x_left = bp.math.reshape(x, (-1, 1))
    x_right = bp.math.repeat(x.reshape((1, -1)), len(x), axis=0)
    d = self.dist(x_left - x_right)
    Jxx = self.J0 * bp.math.exp(-0.5 * bp.math.square(d / self.a)) / \
          (bp.math.sqrt(2 * bp.math.pi) * self.a)
    return Jxx

  def get_stimulus_by_pos(self, pos):
    return self.A * bp.math.exp(-0.25 * bp.math.square(self.dist(self.x - pos) / self.a))

  @staticmethod
  @bp.odeint(method='rk4', dt=0.05)
  def int_u(u, t, conn, k, tau, Iext):
    r1 = bp.math.square(u)
    r2 = 1.0 + k * bp.math.sum(r1)
    r = r1 / r2
    Irec = bp.math.dot(conn, r)
    du = (-u + Irec + Iext) / tau
    return du

  def update(self, _t, _i):
    self.u[:] = self.int_u(self.u, _t, self.conn_mat, self.k, self.tau, self.input)
    self.input[:] = 0.


def task1_population_coding():
  cann = bp.math.jit(CANN1D(num=512, k=0.1, monitors=['u']))

  I1 = cann.get_stimulus_by_pos(0.)
  Iext, duration = bp.inputs.section_input(values=[0., I1, 0.],
                                           durations=[1., 8., 8.],
                                           return_length=True)
  cann.run(duration=duration, inputs=('input', Iext, 'iter'))

  bp.visualize.animate_1D(
    dynamical_vars=[{'ys': cann.mon.u, 'xs': cann.x, 'legend': 'u'},
                    {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
    frame_step=1,
    frame_delay=100,
    show=True,
    # save_path='../../images/CANN-encoding.gif'
  )


def task2_template_matching():
  cann = bp.math.jit(CANN1D(num=512, k=8.1, monitors=['u']))

  dur1, dur2, dur3 = 10., 30., 0.
  num1 = int(dur1 / bp.math.get_dt())
  num2 = int(dur2 / bp.math.get_dt())
  num3 = int(dur3 / bp.math.get_dt())
  Iext = bp.math.zeros((num1 + num2 + num3,) + cann.size)
  Iext[:num1] = cann.get_stimulus_by_pos(0.5)
  Iext[num1:num1 + num2] = cann.get_stimulus_by_pos(0.)
  Iext[num1:num1 + num2] += 0.1 * cann.A * bp.math.random.randn(num2, *cann.size)
  cann.run(duration=dur1 + dur2 + dur3, inputs=('input', Iext, 'iter'))

  bp.visualize.animate_1D(
    dynamical_vars=[{'ys': cann.mon.u, 'xs': cann.x, 'legend': 'u'},
                    {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
    frame_step=5,
    frame_delay=50,
    show=True,
    # save_path='../../images/CANN-decoding.gif'
  )


def task3_smooth_tracking():
  cann = bp.math.jit(CANN1D(num=512, k=8.1, monitors=['u']))

  dur1, dur2, dur3 = 20., 20., 20.
  num1 = int(dur1 / bp.math.get_dt())
  num2 = int(dur2 / bp.math.get_dt())
  num3 = int(dur3 / bp.math.get_dt())
  position = bp.math.zeros(num1 + num2 + num3)
  position[num1: num1 + num2] = bp.math.linspace(0., 12., num2)
  position[num1 + num2:] = 12.
  position = position.reshape((-1, 1))
  Iext = cann.get_stimulus_by_pos(position)
  cann.run(duration=dur1 + dur2 + dur3, inputs=('input', Iext, 'iter'))

  bp.visualize.animate_1D(
    dynamical_vars=[{'ys': cann.mon.u, 'xs': cann.x, 'legend': 'u'},
                    {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
    frame_step=5,
    frame_delay=50,
    show=True,
    # save_path='../../images/CANN-tracking.gif'
  )


task1_population_coding()
task2_template_matching()
task3_smooth_tracking()
