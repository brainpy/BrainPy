import brainpy as bp
import brainpy.math as bm

a, b = 0.02, 0.20


class IzhiJoint(bp.dyn.NeuGroup):
	def dV(self, V, t, u, Iext):
		return 0.04 * V * V + 5 * V + 140 - u + Iext

	def du(self, u, t, V):
		return a * (b * V - u)

	@property
	def derivative(self):
		return bp.JointEq([self.dV, self.du])

	def __init__(self, size):
		super().__init__(size)

		self.V = bm.Variable(bm.zeros(self.num) - 70.)
		self.u = bm.Variable(bm.zeros(self.num))
		self.input = bm.Variable(bm.zeros(self.num))

		self.integral = bp.odeint(self.derivative, method='rk2')

	def update(self, t, dt):
		V, u = self.integral(self.V, self.u, t, self.input, dt=dt)
		spike = V >= 0.
		self.V.value = bm.where(spike, -65., V)
		self.u.value = bm.where(spike, u + 8., u)
		self.input[:] = 0.


class IzhiSeparate(bp.NeuGroup):
	def dV(self, V, t, u, Iext):
		return 0.04 * V * V + 5 * V + 140 - u + Iext

	def du(self, u, t, V):
		return a * (b * V - u)

	def __init__(self, size):
		super().__init__(size)

		self.V = bm.Variable(bm.zeros(self.num) - 70.)
		self.u = bm.Variable(bm.zeros(self.num))
		self.input = bm.Variable(bm.zeros(self.num))

		self.int_V = bp.odeint(self.dV, method='rk2')
		self.int_u = bp.odeint(self.du, method='rk2')

	def update(self, t, dt):
		V = self.int_V(self.V, t, self.u, self.input, dt=dt)
		u = self.int_u(self.u, t, self.V, dt=dt)
		spike = V >= 0.
		self.V.value = bm.where(spike, -65., V)
		self.u.value = bm.where(spike, u + 8., u)
		self.input[:] = 0.


neu1 = IzhiJoint(1)
runner = bp.StructRunner(neu1, monitors=['V'], inputs=('input', 20.), dt=0.2)
runner(800)
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, alpha=0.6, legend='V - joint', show=False)

neu2 = IzhiSeparate(1)
runner = bp.StructRunner(neu2, monitors=['V'], inputs=('input', 20.), dt=0.2)
runner(800)
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, alpha=0.6, legend='V - separate', show=True)
