# -*- coding: utf-8 -*-

from brainpy import math as bm, dyn
from brainpy.integrators import odeint, JointEq, IntegratorRunner

__all__ = [
  'henon_map_series',
  'logistic_map_series',
  'modified_lu_chen_series',
  'mackey_glass_series',

  'rabinovich_fabrikant_series',
  'chen_chaotic_series',
  'lu_chen_chaotic_series',
  'chua_chaotic_series',
  'modified_chua_series',
  'lorenz_series',
  'modified_Lorenz_series',

  'PWL_duffing_series',
]


class HenonMap(dyn.DynamicalSystem):
  """Hénon map."""

  def __init__(self, num, a=1.4, b=0.3):
    super(HenonMap, self).__init__()

    # parameters
    self.a = a
    self.b = b
    self.num = num

    # variables
    self.x = bm.Variable(bm.zeros(num))
    self.y = bm.Variable(bm.zeros(num))

  def update(self, _t, _dt):
    x_new = 1 - self.a * self.x * self.x + self.y
    self.y.value = self.b * self.x
    self.x.value = x_new


def henon_map_series(num_step, a=1.4, b=0.3, inits=None):
  """The Hénon map time series.

  .. math::

    \begin{split}\begin{cases}x_{n+1} = 1-a x_n^2 + y_n\\y_{n+1} = b x_n.\end{cases}\end{split}
  """
  if inits is None:
    inits = {'x': bm.zeros(1), 'y': bm.zeros(1)}
  elif isinstance(inits, dict):
    assert 'x' in inits
    assert 'y' in inits
    inits = {'x': bm.asarray(inits['x']), 'y': bm.asarray(inits['y'])}
    assert inits['x'].shape == inits['y'].shape
  else:
    raise ValueError
  map = HenonMap(inits['x'].size, a=a, b=b)
  runner = dyn.DSRunner(map, monitors=['x', 'y'], dt=1, progress_bar=False)
  runner.run(num_step)
  return {'ts': runner.mon.ts,
          'x': runner.mon.x,
          'y': runner.mon.y}


class LogisticMap(dyn.DynamicalSystem):
  def __init__(self, num, mu=3.):
    super(LogisticMap, self).__init__()

    self.mu = mu
    self.x = bm.Variable(bm.ones(num) * 0.2)

  def update(self, _t, _dt):
    self.x.value = self.mu * self.x * (1 - self.x)


def logistic_map_series(num_step, mu=3., inits=None):
  """The logistic map time series."""
  if inits is None:
    inits = bm.ones(1) * 0.2
  else:
    inits = bm.asarray(inits)
  runner = dyn.DSRunner(LogisticMap(inits.size, mu=mu), monitors=['x'], dt=1, progress_bar=False)
  runner.run(num_step)
  return {'ts': runner.mon.ts, 'x': runner.mon.x}


class ModifiedLuChenSystem(dyn.DynamicalSystem):
  def __init__(self, num, a=35, b=3, c=28, d0=1, d1=1, d2=0., tau=.2, dt=0.1, method='rk4'):
    super(ModifiedLuChenSystem, self).__init__()

    # parameters
    self.a = a
    self.b = b
    self.c = c
    self.d0 = d0
    self.d1 = d1
    self.d2 = d2
    self.tau = tau

    # variables
    self.z_delay = dyn.ConstantDelay(num, delay=tau, dt=dt)
    self.z_delay.data[:] = 14
    self.z = bm.Variable(self.z_delay.latest)
    self.x = bm.Variable(bm.ones(num))
    self.y = bm.Variable(bm.ones(num))

    # functions
    def derivative(x, y, z, t):
      dx = self.a * (y - x)
      f = self.d0 * z + self.d1 * self.z_delay.pull() - self.d2 * bm.sin(self.z_delay.pull())
      dy = (self.c - self.a) * x - x * f + self.c * y
      dz = x * y - self.b * z
      return dx, dy, dz

    self.integral = odeint(derivative, method=method)

  def update(self, _t, _dt):
    self.x.value, self.y.value, self.z.value = self.integral(
      self.x, self.y, self.z, _t, _dt)
    self.z_delay.push(self.z)
    self.z_delay.update(_t, _dt)


def modified_lu_chen_series(duration, dt=0.001, a=36, c=20, b=3, d1=1, d2=0., tau=.2, method='rk4', inits=None):
  if inits is None:
    inits = {'x': bm.ones(1), 'y': bm.ones(1), 'z': bm.ones(1) * 14}
  elif isinstance(inits, dict):
    assert 'x' in inits
    assert 'y' in inits
    assert 'z' in inits
    inits = {'x': bm.asarray(inits['x']),
             'y': bm.asarray(inits['y']),
             'z': bm.asarray(inits['z'])}
    assert inits['x'].shape == inits['y'].shape == inits['z'].shape
  else:
    raise ValueError
  eq = ModifiedLuChenSystem(num=inits['x'].size, a=a, b=b, c=c, d1=d1, d2=d2, tau=tau, dt=dt, method=method)
  eq.x[:] = inits['x']
  eq.y[:] = inits['y']
  eq.z[:] = inits['z']
  runner = dyn.DSRunner(eq, monitors=['x', 'y', 'z'], dt=dt, progress_bar=False)
  runner.run(duration)
  return {'ts': runner.mon.ts,
          'x': runner.mon['x'],
          'y': runner.mon['y'],
          'z': runner.mon['z']}


class MackeyGlassEq(dyn.DynamicalSystem):
  r"""The Mackey-Glass equation is the nonlinear time delay differential equation.

  .. math::

     \frac{dP(t)}{dt} = \frac{\beta P(t - \tau)}{1 + P(t - \tau)^n} - \gamma P(t)

  where $\beta = 0.2$, $\gamma = 0.1$, $n = 10$, and the time delay $\tau = 17$. $\tau$
  controls the chaotic behaviour of the equations (the higher it is, the more chaotic
  the timeserie becomes.)

  - Copied from https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/mackey_glass_eq.html

  """

  def __init__(self, num, inits, beta=2., gamma=1., tau=2., n=9.65, method='rk4', seed=None):
    super(MackeyGlassEq, self).__init__()

    # parameters
    self.beta = beta
    self.gamma = gamma
    self.tau = tau
    self.n = n

    # variables
    rng = bm.random.RandomState(seed)
    self.x = dyn.ConstantDelay(num, delay=tau)
    self.x.data[:] = inits + 0.2 * (rng.random(num) - 0.5)
    self.x_latest = bm.Variable(self.x.latest)
    self.x_oldest = bm.Variable(self.x.oldest)

    # functions
    self.derivative = lambda x, t, x_tau: self.beta * x_tau / (1 + x_tau ** self.n) - self.gamma * x
    self.integral = odeint(self.derivative, method=method)

  def update(self, _t, _dt):
    self.x_oldest.value = self.x.pull()
    self.x_latest.value = self.integral(self.x_latest, _t, self.x_oldest, _dt)
    self.x.push(self.x_latest)
    self.x.update(_t, _dt)


def mackey_glass_series(duration, dt=0.1, beta=2., gamma=1., tau=2., n=9.65,
                        inits=None, method='rk4', seed=None):
  """The Mackey-Glass time series.

  Its dynamics is governed by

  .. math::

     \frac{dP(t)}{dt} = \frac{\beta P(t - \tau)}{1 + P(t - \tau)^n} - \gamma P(t)

  where $\beta = 0.2$, $\gamma = 0.1$, $n = 10$, and the time delay $\tau = 17$. $\tau$
  controls the chaotic behaviour of the equations (the higher it is, the more chaotic
  the timeserie becomes.)

  Parameters
  ----------
  duration: int
  dt: float, int, optional
  beta: float, JaxArray
  gamma: float, JaxArray
  tau: float, JaxArray
  n: float, JaxArray
  inits: optional, float, JaxArray
  method: str
  seed: optional, int

  Returns
  -------
  result: dict

  """
  if inits is None:
    inits = bm.ones(1) * 1.2
  else:
    inits = bm.asarray(inits)
  eq = MackeyGlassEq(num=inits.size, beta=beta, gamma=gamma, tau=tau, n=n,
                     inits=inits, method=method, seed=seed)
  runner = dyn.DSRunner(eq, monitors=['x_latest', 'x_oldest'], dt=dt, progress_bar=False)
  runner.run(duration)
  return {'ts': runner.mon.ts,
          'x': runner.mon['x_latest'],
          'x_tau': runner.mon['x_oldest']}


def _three_variable_model(integrator, duration, default_inits, inits=None, args=None, dyn_args=None, dt=0.001):
  if inits is None:
    inits = default_inits  # {'x': -1, 'y': 0, 'z': 0.5}
  elif isinstance(inits, dict):
    assert 'x' in inits
    assert 'y' in inits
    assert 'z' in inits
    inits = {'x': bm.asarray(inits['x']),
             'y': bm.asarray(inits['y']),
             'z': bm.asarray(inits['z'])}
    assert inits['x'].shape == inits['y'].shape == inits['z'].shape
  else:
    raise ValueError

  runner = IntegratorRunner(integrator, monitors=['x', 'y', 'z'], inits=inits,
                            args=args, dyn_args=dyn_args, dt=dt, progress_bar=False)
  runner.run(duration)
  return {'ts': runner.mon.ts,
          'x': runner.mon.x,
          'y': runner.mon.y,
          'z': runner.mon.z}


def lorenz_series(duration, dt=0.001, sigma=10, beta=8 / 3, rho=28, method='rk4', inits=None):
  dx = lambda x, t, y: sigma * (y - x)
  dy = lambda y, t, x, z: x * (rho - z) - y
  dz = lambda z, t, x, y: x * y - beta * z
  integral = odeint(JointEq([dx, dy, dz]), method=method)

  return _three_variable_model(integral,
                               default_inits={'x': 8, 'y': 1, 'z': 1},
                               duration=duration, dt=dt, inits=inits)


def rabinovich_fabrikant_series(duration, dt=0.001, alpha=1.1, gamma=0.803, method='rk4', inits=None):
  @odeint(method=method)
  def rf_eqs(x, y, z, t):
    dx = y * (z - 1 + x * x) + gamma * x
    dy = x * (3 * z + 1 - x * x) + gamma * y
    dz = -2 * z * (alpha + x * y)
    return dx, dy, dz

  return _three_variable_model(rf_eqs,
                               default_inits={'x': -1, 'y': 0, 'z': 0.5},
                               duration=duration, dt=dt, inits=inits)


def chen_chaotic_series(duration, dt=0.001, a=40, b=3, c=28, method='euler', inits=None):
  @odeint(method=method)
  def chen_system(x, y, z, t):
    dx = a * (y - x)
    dy = (c - a) * x - x * z + c * y
    dz = x * y - b * z
    return dx, dy, dz

  return _three_variable_model(chen_system,
                               default_inits=dict(x=-0.1, y=0.5, z=-0.6),
                               duration=duration, dt=dt, inits=inits)


def lu_chen_chaotic_series(duration, dt=0.001, a=36, c=20, b=3, u=-15.15, method='rk4', inits=None):
  @odeint(method=method)
  def lu_chen_system(x, y, z, t):
    dx = a * (y - x)
    dy = x - x * z + c * y + u
    dz = x * y - b * z
    return dx, dy, dz

  return _three_variable_model(lu_chen_system,
                               default_inits=dict(x=0.1, y=0.3, z=-0.6),
                               duration=duration, dt=dt, inits=inits)


def chua_chaotic_series(duration, dt=0.001, alpha=10, beta=14.514, gamma=0, a=-1.197, b=-0.646, method='rk4',
                        inits=None):
  @odeint(method=method)
  def chua_equation(x, y, z, t):
    fx = b * x + 0.5 * (a - b) * (bm.abs(x + 1) - bm.abs(x - 1))
    dx = alpha * (y - x) - alpha * fx
    dy = x - y + z
    dz = -beta * y - gamma * z
    return dx, dy, dz

  return _three_variable_model(chua_equation,
                               default_inits=dict(x=0.001, y=0, z=0.),
                               duration=duration, dt=dt, inits=inits)


def modified_chua_series(duration, dt=0.001, alpha=10.82, beta=14.286, a=1.3, b=.11, d=0, method='rk4', inits=None):
  @odeint(method=method)
  def modified_chua_system(x, y, z, t):
    dx = alpha * (y + b * bm.sin(bm.pi * x / 2 / a + d))
    dy = x - y + z
    dz = -beta * y
    return dx, dy, dz

  return _three_variable_model(modified_chua_system,
                               default_inits=dict(x=1, y=1, z=0.),
                               duration=duration, dt=dt, inits=inits)


def modified_Lorenz_series(duration, dt=0.001, a=10, b=8 / 3, c=137 / 5, method='rk4', inits=None):
  @odeint(method=method)
  def modified_Lorenz(x, y, z, t):
    temp = 3 * bm.sqrt(x * x + y * y)
    dx = (-(a + 1) * x + a - c + z * y) / 3 + ((1 - a) * (x * x - y * y) + (2 * (a + c - z)) * x * y) / temp
    dy = ((c - a - z) * x - (a + 1) * y) / 3 + (2 * (a - 1) * x * y + (a + c - z) * (x * x - y * y)) / temp
    dz = (3 * x * x * y - y * y * y) / 2 - b * z
    return dx, dy, dz

  return _three_variable_model(modified_Lorenz,
                               default_inits=dict(x=-8, y=4, z=10),
                               duration=duration, dt=dt, inits=inits)


def _two_variable_model(integrator, duration, default_inits, inits=None, args=None, dyn_args=None, dt=0.001):
  if inits is None:
    inits = default_inits
  elif isinstance(inits, dict):
    assert 'x' in inits
    assert 'y' in inits
    inits = {'x': bm.asarray(inits['x']),
             'y': bm.asarray(inits['y'])}
    assert inits['x'].shape == inits['y'].shape
  else:
    raise ValueError

  runner = IntegratorRunner(integrator, monitors=['x', 'y'], inits=inits,
                            args=args, dyn_args=dyn_args, dt=dt, progress_bar=False)
  runner.run(duration)
  return {'ts': runner.mon.ts,
          'x': runner.mon.x,
          'y': runner.mon.y}


def PWL_duffing_series(duration, dt=0.001, e=0.25, m0=-0.0845, m1=0.66, omega=1, i=-14, method='rk4', inits=None):
  gamma = 0.14 + i / 20

  @odeint(method=method)
  def PWL_duffing_eq(x, y, t):
    dx = y
    dy = -m1 * x - (0.5 * (m0 - m1)) * (abs(x + 1) - abs(x - 1)) - e * y + gamma * bm.cos(omega * t)
    return dx, dy

  return _two_variable_model(PWL_duffing_eq,
                             default_inits=dict(x=0, y=0.),
                             duration=duration, dt=dt, inits=inits)
