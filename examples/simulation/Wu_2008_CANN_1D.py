# %% [markdown]
# # _(Si Wu, 2008)_ Continuous-attractor Neural Network

# %% [markdown]
# Here we show the implementation of the paper:
#
# - Si Wu, Kosuke Hamaguchi, and Shun-ichi Amari. "Dynamics and computation
#   of continuous attractors." Neural computation 20.4 (2008): 994-1025.
#
# Author:
#
# - Chaoming Wang (chao.brain@qq.com)

# %% [markdown]
# The mathematical equation of the Continuous-Attractor Neural Network (CANN) is given by:
#
# $$\tau \frac{du(x,t)}{dt} = -u(x,t) + \rho \int dx' J(x,x') r(x',t)+I_{ext}$$
#
# $$r(x,t) = \frac{u(x,t)^2}{1 + k \rho \int dx' u(x',t)^2}$$
#
# $$J(x,x') = \frac{1}{\sqrt{2\pi}a}\exp(-\frac{|x-x'|^2}{2a^2})$$
#
# $$I_{ext} = A\exp\left[-\frac{|x-z(t)|^2}{4a^2}\right]$$

# %%
import brainpy as bp
import brainpy.math as bm
bm.set_platform('cpu')


# %%
class CANN1D(bp.NeuGroup):
  def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4.,
               z_min=-bm.pi, z_max=bm.pi, **kwargs):
    super(CANN1D, self).__init__(size=num, **kwargs)

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
    self.x = bm.linspace(z_min, z_max, num)  # The encoded feature values
    self.rho = num / self.z_range  # The neural density
    self.dx = self.z_range / num  # The stimulus density

    # variables
    self.u = bm.Variable(bm.zeros(num))
    self.input = bm.Variable(bm.zeros(num))

    # The connection matrix
    self.conn_mat = self.make_conn(self.x)
    
    # function
    self.integral = bp.odeint(self.derivative)

  def derivative(self, u, t, Iext):
    r1 = bm.square(u)
    r2 = 1.0 + self.k * bm.sum(r1)
    r = r1 / r2
    Irec = bm.dot(self.conn_mat, r)
    du = (-u + Irec + Iext) / self.tau
    return du

  def dist(self, d):
    d = bm.remainder(d, self.z_range)
    d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
    return d

  def make_conn(self, x):
    assert bm.ndim(x) == 1
    x_left = bm.reshape(x, (-1, 1))
    x_right = bm.repeat(x.reshape((1, -1)), len(x), axis=0)
    d = self.dist(x_left - x_right)
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / \
          (bm.sqrt(2 * bm.pi) * self.a)
    return Jxx

  def get_stimulus_by_pos(self, pos):
    return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

  def update(self, _t, _dt):
    self.u[:] = self.integral(self.u, _t, self.input)
    self.input[:] = 0.

# %%
cann = CANN1D(num=512, k=0.1)


# %% [markdown]
# ## Population coding

# %%
I1 = cann.get_stimulus_by_pos(0.)
Iext, duration = bp.inputs.section_input(values=[0., I1, 0.],
                                         durations=[1., 8., 8.],
                                         return_length=True)
runner = bp.StructRunner(cann,
                         inputs=['input', Iext, 'iter'],
                         monitors=['u'],
                         dyn_vars=cann.vars())
runner(duration)
bp.visualize.animate_1D(
  dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
                  {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
  frame_step=1,
  frame_delay=100,
  show=True,
  # save_path='../../images/cann-encoding.gif'
)

# %% [markdown]
# ![](../images/cann-encoding.gif)

# %% [markdown]
# ## Template matching

# %% [markdown]
# The cann can perform efficient population decoding by achieving template-matching.

# %%
cann.k = 8.1

dur1, dur2, dur3 = 10., 30., 0.
num1 = int(dur1 / bm.get_dt())
num2 = int(dur2 / bm.get_dt())
num3 = int(dur3 / bm.get_dt())
Iext = bm.zeros((num1 + num2 + num3,) + cann.size)
Iext[:num1] = cann.get_stimulus_by_pos(0.5)
Iext[num1:num1 + num2] = cann.get_stimulus_by_pos(0.)
Iext[num1:num1 + num2] += 0.1 * cann.A * bm.random.randn(num2, *cann.size)

runner = bp.StructRunner(cann,
                         inputs=('input', Iext, 'iter'),
                         monitors=['u'],
                         dyn_vars=cann.vars())
runner(dur1 + dur2 + dur3)
bp.visualize.animate_1D(
  dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
                  {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
  frame_step=5,
  frame_delay=50,
  show=True,
  # save_path='../../images/cann-decoding.gif'
)

# %% [markdown]
# ![](../images/cann-decoding.gif)

# %% [markdown]
# ## Smooth tracking
#
# The cann can track moving stimulus.

# %%
dur1, dur2, dur3 = 10., 100., 20.
num1 = int(dur1 / bm.get_dt())
num2 = int(dur2 / bm.get_dt())
num3 = int(dur3 / bm.get_dt())
position = bm.zeros(num1 + num2 + num3)
position[num1: num1 + num2] = bm.linspace(0., 20., num2)
position[num1 + num2:] = 20.
position = position.reshape((-1, 1))
Iext = cann.get_stimulus_by_pos(position)
runner = bp.StructRunner(cann,
                         inputs=('input', Iext, 'iter'),
                         monitors=['u'],
                         dyn_vars=cann.vars())
runner(dur1 + dur2 + dur3)
bp.visualize.animate_1D(
  dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
                  {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
  frame_step=5,
  frame_delay=50,
  show=True,
  save_path='./cann-tracking.mp4'
)

# %% [markdown]
# ![](../images/cann-tracking.gif)
