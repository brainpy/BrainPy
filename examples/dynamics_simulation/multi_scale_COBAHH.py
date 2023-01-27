# -*- coding: utf-8 -*-

import os.path

import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.channels import INa_TM1991, IL
from brainpy.synapses import Exponential
from brainpy.synouts import COBA
from brainpy.connect import FixedProb
from jax import vmap

comp_method = 'sparse'


area_names = ['V1', 'V2', 'V4', 'TEO', 'TEpd']
data = np.load('./data/visual_conn.npz')
conn_data = data['conn']
delay_data = (data['delay'] / bm.get_dt()).astype(int)
num_exc = 3200
num_inh = 800


class IK(bp.Channel):
  def __init__(self, size, E=-90., g_max=10., phi=1., V_sh=-50.):
    super(IK, self).__init__(size)
    self.g_max, self.E, self.V_sh, self.phi = g_max, E, V_sh, phi
    self.p = bm.Variable(bm.zeros(size))
    self.integral = bp.odeint(self.dp, method='exp_euler')

  def dp(self, p, t, V):
    tmp = V - self.V_sh - 15.
    alpha = 0.032 * tmp / (1. - bm.exp(-tmp / 5.))
    beta = 0.5 * bm.exp(-(V - self.V_sh - 10.) / 40.)
    return self.phi * (alpha * (1. - p) - beta * p)

  def update(self, tdi, V):
    self.p.value = self.integral(self.p, tdi.t, V, dt=tdi.dt)

  def current(self, V):
    return self.g_max * self.p ** 4 * (self.E - V)


class HH(bp.CondNeuGroup):
  def __init__(self, size):
    super(HH, self).__init__(size, V_initializer=bp.init.Uniform(-70, -50.))
    self.IK = IK(size, g_max=30., V_sh=-63.)
    self.INa = INa_TM1991(size, g_max=100., V_sh=-63.)
    self.IL = IL(size, E=-60., g_max=0.05)


class Network(bp.Network):
  def __init__(self, num_E, num_I, gEE=0.03, gEI=0.03, gIE=0.335, gII=0.335):
    super(Network, self).__init__()
    self.E, self.I = HH(num_E), HH(num_I)
    self.E2E = Exponential(self.E, self.E, FixedProb(0.02),
                           g_max=gEE,
                           tau=5,
                           output=COBA(E=0.),
                           comp_method=comp_method)
    self.E2I = Exponential(self.E, self.I, FixedProb(0.02),
                           g_max=gEI,
                           tau=5.,
                           output=COBA(E=0.),
                           comp_method=comp_method)
    self.I2E = Exponential(self.I, self.E, FixedProb(0.02),
                           g_max=gIE,
                           tau=10.,
                           output=COBA(E=-80),
                           comp_method=comp_method)
    self.I2I = Exponential(self.I, self.I, FixedProb(0.02),
                           g_max=gII,
                           tau=10.,
                           output=COBA(E=-80.),
                           comp_method=comp_method)


class Projection(bp.DynamicalSystem):
  def __init__(self, pre, post, delay, conn, gEE=0.03, gEI=0.03, tau=5.):
    super(Projection, self).__init__()
    self.pre = pre
    self.post = post
    self.E2E = Exponential(pre.E, post.E, bp.conn.FixedProb(0.02),
                           delay_step=delay,
                           g_max=gEE * conn,
                           tau=tau,
                           output=COBA(0.),
                           comp_method=comp_method)
    self.E2I = Exponential(pre.E, post.I, bp.conn.FixedProb(0.02),
                           delay_step=delay,
                           g_max=gEI * conn,
                           tau=tau,
                           output=COBA(0.),
                           comp_method=comp_method)

  def update(self, tdi):
    self.E2E.update(tdi)
    self.E2I.update(tdi)


class System(bp.Network):
  def __init__(self, conn, delay, gEE=0.03, gEI=0.03, gIE=0.335, gII=0.335):
    super(System, self).__init__()

    num_area = conn.shape[0]
    self.areas = [Network(num_exc, num_inh, gEE=gEE, gEI=gEI, gII=gII, gIE=gIE)
                  for _ in range(num_area)]
    self.projections = []
    for i in range(num_area):
      for j in range(num_area):
        if i != j:
          proj = Projection(self.areas[j],
                            self.areas[i],
                            delay=delay[i, j],
                            conn=conn[i, j],
                            gEE=gEE,
                            gEI=gEI)
          self.projections.append(proj)
    self.register_implicit_nodes(self.projections, self.areas)


def single_run(gc, gEE, gEI, gIE, gII, inputs, duration, seed=123, save_fig=False):
  bm.random.seed(seed)
  circuit = System(gc * bm.asarray(conn_data),
                   bm.asarray(delay_data),
                   gEE=gEE, gEI=gEI, gIE=gIE, gII=gII)
  f1 = lambda tdi: bm.concatenate([area.E.spike for area in circuit.areas])
  f2 = lambda tdi: bm.concatenate([area.I.spike for area in circuit.areas])
  runner = bp.DSRunner(
    circuit,
    fun_monitors={'exc.spike': f1, 'inh.spike': f2},
    inputs=[circuit.areas[0].E.input, inputs, 'iter'],
    numpy_mon_after_run=False
  )
  runner.run(duration)
  fig, gs = bp.visualize.get_figure(5, 1, 2, 10)
  fig.add_subplot(gs[0:2, 0])
  bp.visualize.raster_plot(runner.mon['ts'], runner.mon.get('exc.spike'))
  plt.title(f'gc={gc}, gEE={gEE}, gEI={gEI}, gIE={gIE}, gII={gII}, seed={seed}')
  fig.add_subplot(gs[2:4, 0])
  bp.visualize.raster_plot(runner.mon['ts'], runner.mon.get('inh.spike'))
  fig.add_subplot(gs[4, 0])
  plt.plot(runner.mon['ts'], bm.as_numpy(inputs))
  plt.ylabel('Current')
  plt.tight_layout()
  if save_fig:
    plt.savefig(f'results/{seed}.png')
  else:
    plt.show()
  plt.close(fig)


def vmap_search(gc=3., I_size=0.2, I_duration=400., e_range=(0.1, 1.1, 0.1), i_range=(0.1, 1.1, 0.1)):
  I, duration = bp.inputs.section_input([0, I_size, 0.], [200., I_duration, 200.], return_length=True)
  e_scale = bm.arange(*e_range)
  i_scale = bm.arange(*i_range)

  path = (f'results_comp={comp_method}_gc={gc}'
          f'_I={I_size}_Ilength={I_duration}_'
          f'escale={e_range[0]}-{e_range[1]}_'
          f'iscale={i_range[0]}-{i_range[1]}')
  if not os.path.exists(path):
    os.makedirs(path)
  else:
    raise ValueError(f'The directory has been existed: {path}')

  @vmap
  def run(gE, gI):
    bm.random.seed(123)
    circuit = System(bm.asarray(conn_data) * gc,
                     bm.asarray(delay_data),
                     gE=gE, gI=gI)
    f1 = lambda tdi: bm.concatenate([area.E.spike for area in circuit.areas])
    f2 = lambda tdi: bm.concatenate([area.I.spike for area in circuit.areas])
    runner = bp.DSRunner(
      circuit,
      fun_monitors={'exc.spike': f1, 'inh.spike': f2},
      inputs=[circuit.areas[0].E.input, I, 'iter'],
      numpy_mon_after_run=False
    )
    runner.run(duration)
    runner.mon.pop('var_names')
    return runner.mon

  ee_scale, ii_scale = bm.meshgrid(e_scale, i_scale)
  ee_weights = ee_scale.flatten() * 0.03
  ii_weights = ii_scale.flatten() * 0.335

  monitors = run(ee_weights, ii_weights)
  monitors.to_numpy()

  for i, (ge, gi) in enumerate(zip(bm.as_numpy(ee_weights), bm.as_numpy(ii_weights))):
    name = f'gE={ge:.5f}, gI={gi:.5f}'
    fig, gs = bp.visualize.get_figure(5, 1, 2, 10)
    fig.add_subplot(gs[0:2, 0])
    bp.visualize.raster_plot(monitors['ts'][i], monitors.get('exc.spike')[i])
    plt.title(name)
    fig.add_subplot(gs[2:4, 0])
    bp.visualize.raster_plot(monitors['ts'][i], monitors.get('inh.spike')[i])
    fig.add_subplot(gs[4, 0])
    plt.plot(monitors['ts'][i], bm.as_numpy(I))
    plt.ylabel('Current')
    fn_name = f'{path}/{name}.png'
    print(f'Saving {fn_name} ...')
    plt.tight_layout()
    plt.savefig(fn_name)
    plt.close(fig)

  bm.clear_buffer_memory()


# single_run(0.006, 0.1675, *bp.inputs.section_input([0, 0.8], [200., 400.], return_length=True))

# single_run(2., 0.003, 0.2345, *bp.inputs.section_input([0., 0.2, 0.], [200., 400., 300.], return_length=True),
#            seed=12345)

# single_run(1., 0.0030, 0.3350, *bp.inputs.section_input([0., 0.8, 0.], [200., 400., 300.], return_length=True),
#            seed=None)

def run_one_seed():
  single_run(
    1., 0.0060, 0.0060, 0.26800, 0.26800,
    *bp.inputs.section_input([0., 1., 0.],
                             [400., 100., 800.],
                             return_length=True),
    seed=20873
  )


def search_seeds():
  for _ in range(100):
    s = bp.tools.format_seed()
    print(s)
    single_run(
      1., 0.0060, 0.0060, 0.26800, 0.26800,
      *bp.inputs.section_input([0., 1., 0.],
                               [400., 100., 300.],
                               return_length=True),
      seed=s
    )
    bm.clear_buffer_memory()


def visualize(seed=20873, gc=1., gEE=0.0060, gEI=0.0060, gIE=.26800, gII=0.26800, ):
  bm.random.seed(seed)
  model = System(gc * bm.asarray(conn_data), bm.asarray(delay_data),
                 gEE=gEE, gEI=gEI, gIE=gIE, gII=gII)
  inputs, duration = bp.inputs.section_input([0., 1., 0.],
                                             [400., 100., 300.],
                                             return_length=True)
  runner = bp.DSRunner(
    model,
    fun_monitors={
      'exc.spike': lambda tdi: bm.concatenate([area.E.spike for area in model.areas]),
      'inh.spike': lambda tdi: bm.concatenate([area.I.spike for area in model.areas]),
      'V1.E.V': lambda tdi: model.areas[0].E.spike,
      'V1.I.V': lambda tdi: model.areas[0].I.spike,
      'V2.E.V': lambda tdi: model.areas[1].I.spike,
      'V1.E.K.p': lambda tdi: model.areas[0].E.IK.p,
    },
    inputs=[model.areas[0].E.input, inputs, 'iter'],
    numpy_mon_after_run=False
  )
  runner.run(duration)

  fig, gs = bp.visualize.get_figure(5, 1, 2, 10)
  fig.add_subplot(gs[0:2, 0])
  bp.visualize.raster_plot(runner.mon['ts'], runner.mon.get('exc.spike'))
  plt.title(f'gc={gc}, gEE={gEE}, gEI={gEI}, gIE={gIE}, gII={gII}, seed={seed}')
  fig.add_subplot(gs[2:4, 0])
  bp.visualize.raster_plot(runner.mon['ts'], runner.mon.get('inh.spike'))
  fig.add_subplot(gs[4, 0])
  plt.plot(runner.mon['ts'], bm.as_numpy(inputs))
  plt.ylabel('Current')
  plt.show()


if __name__ == '__main__':
  # run_one_seed()

  seed = 1824455  # 666233 # 20873
  seed = 2546234
  # seed = 4287332
  gc = 1.
  gEE = 0.0060
  gEI = 0.0060
  gIE = 0.26800
  gII = 0.26800

  bm.random.seed(seed)
  model = System(gc * bm.asarray(conn_data), bm.asarray(delay_data),
                 gEE=gEE, gEI=gEI, gIE=gIE, gII=gII)
  inputs, duration = bp.inputs.section_input([0., 1., 0.],
                                             [400., 100., 300.],
                                             return_length=True)
  runner = bp.DSRunner(
    model,
    monitors={
      'exc.spike': lambda tdi: bm.concatenate([area.E.spike for area in model.areas]),
      'inh.spike': lambda tdi: bm.concatenate([area.I.spike for area in model.areas]),
      'V1.E.V': lambda tdi: model.areas[0].E.V,
      'V1.E.spike': lambda tdi: model.areas[0].E.spike,
      'V1.I.spike': lambda tdi: model.areas[0].I.spike,
      # 'V2.E.V': lambda tdi: model.areas[1].I.V,
      'V1.E.K.p': lambda tdi: model.areas[0].E.IK.p,
    },
    inputs=[model.areas[0].E.input, inputs, 'iter'],
  )
  runner.run(duration)

  # visualization
  # fig, gs = bp.visualize.get_figure(5, 1, 2, 10)
  # fig.add_subplot(gs[0:2, 0])
  # bp.visualize.raster_plot(runner.mon['ts'], runner.mon.get('exc.spike'))
  # plt.title(f'gc={gc}, gEE={gEE}, gEI={gEI}, gIE={gIE}, gII={gII}, seed={seed}')
  # fig.add_subplot(gs[2:4, 0])
  # bp.visualize.raster_plot(runner.mon['ts'], runner.mon.get('inh.spike'))
  # fig.add_subplot(gs[4, 0])
  # plt.plot(runner.mon['ts'], bm.as_numpy(inputs))
  # plt.ylabel('Current')
  # plt.show()

  fig, gs = bp.visualize.get_figure(2, 1, 2.25 * 1, 6 * 1)
  plot_ids = [0, 2, 4, 8]
  fig.add_subplot(gs[0, 0])
  for i in plot_ids:
    plt.plot(runner.mon['ts'], runner.mon.get('V1.E.K.p')[:, i])
  plt.ylabel(r'$p$')
  plt.xticks([])
  plt.yticks([])
  plt.xlim(0, 800.)
  plt.title('Channel and Neuron')
  fig.add_subplot(gs[1, 0])
  for i in plot_ids:
    plt.plot(runner.mon['ts'], runner.mon.get('V1.E.V')[:, i])
  plt.xlabel('Time [ms]')
  plt.ylabel(r'$V$')
  plt.yticks([])
  plt.xlim(0, 800.)
  plt.show()


  # V1 raster plot and firing rate
  fig, gs = bp.visualize.get_figure(3, 1, 1.5, 6.)
  fig.add_subplot(gs[0: 2, 0])
  indices, times = bp.measure.raster_plot(runner.mon['V1.E.spike'], runner.mon['ts'])
  plt.plot(times, indices, '.', markersize=1)
  plt.xticks([])
  plt.yticks([])
  plt.ylabel('Raster Plot')
  plt.title('V1 Network')
  fig.add_subplot(gs[2, 0])
  rate = bp.measure.firing_rate(runner.mon['V1.E.spike'], 20.)
  plt.plot(runner.mon['ts'], rate)
  plt.yticks([])
  plt.xlabel('Time [ms]')
  plt.ylabel('Firing Rate')
  plt.show()

  # Whole network raster plot and firing rate
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  fig.add_subplot(gs[0, 0])
  indices, times = bp.measure.raster_plot(runner.mon['exc.spike'], runner.mon['ts'])
  plt.plot(times, indices, '.', markersize=1)
  plt.xlim(375., 750.)
  plt.ylim(0, len(area_names) * num_exc)
  plt.yticks(np.arange(len(area_names)) * num_exc + num_exc / 2, area_names)
  plt.plot([375., 750.], (np.arange(len(area_names) + 1) * num_exc).repeat(2).reshape(-1, 2).T, 'k-')
  plt.title('Visual System')
  plt.xlabel('Time [ms]')
  plt.show()

