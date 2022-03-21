# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.integrators import odeint, sdeint, JointEq
from brainpy.types import Parameter, Shape
from brainpy.tools.checking import check_float
from .base import RateModel

__all__ = [
  'FHN'
]


class FHN(RateModel):
  r"""FitzHugh-Nagumo system used in [1]_.

  .. math::

     \frac{dx}{dt} = -\alpha V^3 + \beta V^2 + \gamma V - w + I_{ext}\\
     \tau \frac{dy}{dt} = (V - \delta  - \epsilon w)

  Parameters
  ----------
  size: Shape
    The model size.

  coupling: str
    The way of coupling.
  gc: float
    The global coupling strength.
  signal_speed: float
    Signal transmission speed between areas.
  sc_mat: optional, tensor
    Structural connectivity matrix. Adjacency matrix of coupling strengths,
    will be normalized to 1. If not given, then a single node simulation
    will be assumed. Default None
  fl_mat: optional, tensor
    Fiber length matrix. Will be used for computing the
    delay matrix together with the signal transmission
    speed parameter `signal_speed`. Default None.

  References
  ----------
  .. [1] Kostova, T., Ravindran, R., & Schonbek, M. (2004). FitzHugh–Nagumo
         revisited: Types of bifurcations, periodical forcing and stability
         regions by a Lyapunov functional. International journal of
         bifurcation and chaos, 14(03), 913-925.

  """

  def __init__(self,
               size: Shape,

               # fhn parameters
               alpha: Parameter = 3.0,
               beta: Parameter = 4.0,
               gamma: Parameter = -1.5,
               delta: Parameter = 0.0,
               epsilon: Parameter = 0.5,
               tau: Parameter = 20.0,

               # noise parameters
               x_ou_mean: Parameter = 0.0,
               y_ou_mean: Parameter = 0.0,
               ou_sigma: Parameter = 0.0,
               ou_tau: Parameter = 5.0,

               # coupling parameters
               coupling: str = 'diffusive',
               gc=0.6,
               signal_speed=20.0,
               sc_mat=None,
               fl_mat=None,

               # other parameters
               method: str = None,
               name: str = None):
    super(FHN, self).__init__(size, name=name)

    # model parameters
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.delta = delta
    self.epsilon = epsilon
    self.tau = tau

    # noise parameters
    self.x_ou_mean = x_ou_mean  # mV/ms, OU process
    self.y_ou_mean = y_ou_mean  # mV/ms, OU process
    self.ou_sigma = ou_sigma  # mV/ms/sqrt(ms), noise intensity
    self.ou_tau = ou_tau  # ms, timescale of the Ornstein-Uhlenbeck noise process

    # coupling parameters
    # ----
    # The coupling parameter determines how nodes are coupled.
    # "diffusive" for diffusive coupling,
    # "additive" for additive coupling
    self.coupling = coupling
    assert coupling in ['diffusive', 'additive'], (f'Only support "diffusive" and "additive" '
                                                   f'coupling, while we got {coupling}')
    check_float(gc, 'gc', allow_none=False, allow_int=False)
    self.gc = gc  # global coupling strength
    check_float(signal_speed, 'signal_speed', allow_none=False, allow_int=True)
    self.signal_speed = signal_speed  # signal transmission speed between areas


    # variables
    self.x = bm.Variable(bm.random.random(self.num) * 0.05)
    self.y = bm.Variable(bm.random.randint(self.num) * 0.05)
    self.x_ou = bm.Variable(bm.ones(self.num) * x_ou_mean)
    self.y_ou = bm.Variable(bm.ones(self.num) * y_ou_mean)
    self.x_ext = bm.Variable(bm.zeros(self.num))
    self.y_ext = bm.Variable(bm.zeros(self.num))

    # integral functions
    self.int_ou = sdeint(f=self.df_ou, g=self.dg_ou, method='euler')
    self.int_xy = odeint(f=JointEq([self.dx, self.dy]), method=method)

  def dx(self, x, t, y, x_ext):
    return - self.alpha * x ** 3 + self.beta * x ** 2 + self.gamma * x - y + x_ext

  def dy(self, y, t, x, y_ext=0.):
    return (x - self.delta - self.epsilon * y + y_ext) / self.tau

  def df_ou(self, x_ou, y_ou, t):
    f_x_ou = (self.x_ou_mean - x_ou) / self.ou_tau
    f_y_ou = (self.y_ou_mean - y_ou) / self.ou_tau
    return f_x_ou, f_y_ou

  def dg_ou(self, x_ou, y_ou, t):
    return self.ou_sigma, self.ou_sigma

  def update(self, _t, _dt):
    x_ext = self.x_ext + self.x_ou
    y_ext = self.y_ext + self.y_ou
    x, y = self.int_xy(self.x, self.y, _t, x_ext=x_ext, y_ext=y_ext, dt=_dt)
    self.x.value = x
    self.y.value = y
    x_ou, y_ou = self.int_ou(self.x_ou, self.y_ou, _t, _dt)
    self.x_ou.value = x_ou
    self.y_ou.value = y_ou
