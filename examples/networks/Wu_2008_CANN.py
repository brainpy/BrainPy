# -*- coding: utf-8 -*-


import numpy as np
import brainpy as bp


class CANN1D(bp.NeuGroup):
    target_backend = ['numpy', 'numba']

    def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4.,
                 z_min=-np.pi, z_max=np.pi, **kwargs):
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
        self.x = np.linspace(z_min, z_max, num)  # The encoded feature values

        # variables
        self.u = np.zeros(num)
        self.input = np.zeros(num)

        # The connection matrix
        self.conn_mat = self.make_conn(self.x)

        super(CANN1D, self).__init__(size=num, **kwargs)

        self.rho = num / self.z_range  # The neural density
        self.dx = self.z_range / num  # The stimulus density

    def dist(self, d):
        d = np.remainder(d, self.z_range)
        d = np.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self, x):
        assert np.ndim(x) == 1
        x_left = np.reshape(x, (-1, 1))
        x_right = np.repeat(x.reshape((1, -1)), len(x), axis=0)
        d = self.dist(x_left - x_right)
        Jxx = self.J0 * np.exp(-0.5 * np.square(d / self.a)) / (np.sqrt(2 * np.pi) * self.a)
        return Jxx

    def get_stimulus_by_pos(self, pos):
        return self.A * np.exp(-0.25 * np.square(self.dist(self.x - pos) / self.a))

    @staticmethod
    @bp.odeint(method='rk4', dt=0.05)
    def int_u(u, t, conn, k, tau, Iext):
        r1 = np.square(u)
        r2 = 1.0 + k * np.sum(r1)
        r = r1 / r2
        Irec = np.dot(conn, r)
        du = (-u + Irec + Iext) / tau
        return du

    def update(self, _t):
        self.u = self.int_u(self.u, _t, self.conn_mat, self.k, self.tau, self.input)
        self.input[:] = 0.


def task1_population_coding():
    cann = CANN1D(num=512, k=0.1, monitors=['u'])

    I1 = cann.get_stimulus_by_pos(0.)
    Iext, duration = bp.inputs.constant_current([(0., 1.), (I1, 8.), (0., 8.)])
    cann.run(duration=duration, inputs=('input', Iext))

    bp.visualize.animate_1D(
        dynamical_vars=[{'ys': cann.mon.u, 'xs': cann.x, 'legend': 'u'},
                        {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
        frame_step=1,
        frame_delay=100,
        show=True,
        # save_path='../../images/CANN-encoding.gif'
    )


def task2_template_matching():
    cann = CANN1D(num=512, k=8.1, monitors=['u'])

    dur1, dur2, dur3 = 10., 30., 0.
    num1 = int(dur1 / bp.backend.get_dt())
    num2 = int(dur2 / bp.backend.get_dt())
    num3 = int(dur3 / bp.backend.get_dt())
    Iext = np.zeros((num1 + num2 + num3,) + cann.size)
    Iext[:num1] = cann.get_stimulus_by_pos(0.5)
    Iext[num1:num1 + num2] = cann.get_stimulus_by_pos(0.)
    Iext[num1:num1 + num2] += 0.1 * cann.A * np.random.randn(num2, *cann.size)
    cann.run(duration=dur1 + dur2 + dur3, inputs=('input', Iext))

    bp.visualize.animate_1D(
        dynamical_vars=[{'ys': cann.mon.u, 'xs': cann.x, 'legend': 'u'},
                        {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
        frame_step=5,
        frame_delay=50,
        show=True,
        # save_path='../../images/CANN-decoding.gif'
    )


def task3_smooth_tracking():
    cann = CANN1D(num=512, k=8.1, monitors=['u'])

    dur1, dur2, dur3 = 20., 20., 20.
    num1 = int(dur1 / bp.backend.get_dt())
    num2 = int(dur2 / bp.backend.get_dt())
    num3 = int(dur3 / bp.backend.get_dt())
    position = np.zeros(num1 + num2 + num3)
    position[num1: num1 + num2] = np.linspace(0., 12., num2)
    position[num1 + num2:] = 12.
    position = position.reshape((-1, 1))
    Iext = cann.get_stimulus_by_pos(position)
    cann.run(duration=dur1 + dur2 + dur3, inputs=('input', Iext))

    bp.visualize.animate_1D(
        dynamical_vars=[{'ys': cann.mon.u, 'xs': cann.x, 'legend': 'u'},
                        {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
        frame_step=5,
        frame_delay=50,
        show=True,
        # save_path='../../images/CANN-tracking.gif'
    )


task3_smooth_tracking()
