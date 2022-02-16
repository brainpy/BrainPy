# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm


__all__ = [
  'mackey_glass_series',
]


def henon_map(num_step, ):
  pass


def logistic_map(num_step):
  pass


def lorenz_series(num_step, dt=0.001):
  pass


class MackeyGlassEq(bp.NeuGroup):
  r"""The Mackey-Glass equation is the nonlinear time delay differential equation.

  .. math::

     \frac{dP(t)}{dt} = \frac{\beta P(t - \tau)}{1 + P(t - \tau)^n} - \gamma P(t)

  where $\beta = 0.2$, $\gamma = 0.1$, $n = 10$, and the time delay $\tau = 17$. $\tau$
  controls the chaotic behaviour of the equations (the higher it is, the more chaotic
  the timeserie becomes.)

  - Copied from https://brainpy-examples.readthedocs.io/en/latest/classical_dynamical_systems/mackey_glass_eq.html

  """
  def __init__(self, num, inits, beta=2., gamma=1., tau=2., n=9.65,
               method='rk4', seed=None):
    super(MackeyGlassEq, self).__init__(num)

    # parameters
    self.beta = beta
    self.gamma = gamma
    self.tau = tau
    self.n = n

    # variables
    rng = bm.random.RandomState(seed)
    self.x = bp.ConstantDelay(num, delay=tau)
    self.x.data[:] = inits + 0.2 * (rng.random(num) - 0.5)
    self.x_latest = bm.Variable(self.x.latest)
    self.x_oldest = bm.Variable(self.x.oldest)

    # functions
    self.derivative = lambda x, t, x_tau: self.beta * x_tau / (1 + x_tau ** self.n) - self.gamma * x
    self.integral = bp.odeint(self.derivative, method=method)

  def update(self, _t, _dt):
    self.x_oldest.value = self.x.pull()
    self.x_latest.value = self.integral(self.x_latest, _t, self.x_oldest, _dt)
    self.x.push(self.x_latest)
    self.x.update(_t, _dt)


def mackey_glass_series(duration, dt=0.1, beta=2., gamma=1., tau=2., n=9.65,
                        inits=None, method='rk4', seed=None):
  if inits is None:
    inits = bm.ones(1) * 1.2
  else:
    inits = bm.asarray(inits)
  eq = MackeyGlassEq(num=inits.size, beta=beta, gamma=gamma, tau=tau, n=n,
                     inits=inits, method=method, seed=seed)
  runner = bp.StructRunner(eq, monitors=['x_latest', 'x_oldest'], dt=dt, progress_bar=False)
  runner.run(duration)
  return {'ts': runner.mon.ts,
          'x': runner.mon['x_latest'],
          'x_tau': runner.mon['x_oldest']}
