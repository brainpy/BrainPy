import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')


class CANN2D(bp.NeuGroup):
  def __init__(self, length, tau=1., k=8.1, a=0.5, A=10., J0=4.,
               z_min=-bm.pi, z_max=bm.pi, name=None):
    super(CANN2D, self).__init__(size=(length, length), name=name)

    # parameters
    self.length = length
    self.tau = tau  # The synaptic time constant
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, length)  # The encoded feature values
    self.rho = length / self.z_range  # The neural density
    self.dx = self.z_range / length  # The stimulus density

    # variables
    self.r = bm.Variable(bm.zeros((length, length)))
    self.u = bm.Variable(bm.zeros((length, length)))
    self.input = bm.Variable(bm.zeros((length, length)))

    # The connection matrix
    self.conn_mat = self.make_conn(self.x)

  def show_conn(self):
    plt.imshow(np.asarray(self.conn_mat))
    plt.colorbar()
    plt.show()

  def dist(self, d):
    v_size = bm.asarray([self.z_range, self.z_range])
    return bm.where(d > v_size / 2, v_size - d, d)

  def make_conn(self, x):
    x1, x2 = bm.meshgrid(x, x)
    value = bm.stack([x1.flatten(), x2.flatten()]).T
    d = self.dist(bm.abs(value[0] - value))
    d = bm.linalg.norm(d, axis=1)
    d = d.reshape((self.length, self.length))
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
    return Jxx

  def get_stimulus_by_pos(self, pos):
    return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

  def update(self, _t, _dt):
    r1 = bm.square(self.u)
    r2 = 1.0 + self.k * bm.sum(r1)
    self.r.value = r1 / r2
    r = bm.fft.fft2(self.r)
    jjft = bm.fft.fft2(self.conn_mat)
    interaction = bm.real(bm.fft.ifft2(r * jjft))
    self.u.value = self.u + (-self.u + self.input + interaction) / self.tau * _dt
    self.input[:] = 0.


cann = CANN2D(length=512, k=0.1)

runner = bp.StructRunner(cann,
                         # inputs=['input', Iext, 'iter'],
                         # monitors=['u'],
                         dyn_vars=cann.vars())
t = runner.run(1000.)
print(t)
