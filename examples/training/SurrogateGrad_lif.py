# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from brainpy.dyn import neurons, synapses, synouts


class RSNN(bp.dyn.Network):
  def __init__(self, num_in, num_rec, num_out):
    super(RSNN, self).__init__()

    # parameters
    self.num_in = num_in
    self.num_rec = num_rec
    self.num_out = num_out

    # neuron groups
    self.i = neurons.InputGroup(num_in, trainable=True)
    self.r = neurons.LIF(num_rec, tau=10, V_reset=0, V_rest=0, V_th=1, tau_ref=0., trainable=True)
    self.o = neurons.LeakyIntegrator(num_out, tau=5, trainable=True)

    # synapses
    self.i2r = synapses.Exponential(self.i, self.r, bp.conn.All2All(), synouts.CUBA(),
                                    tau=10., g_max=bp.init.XavierUniform(), trainable=True)
    self.r2o = synapses.Exponential(self.r, self.o, bp.conn.All2All(), synouts.CUBA(),
                                    tau=10., g_max=bp.init.XavierUniform(), trainable=True)

  def update(self, tdi, spike):
    self.i2r(tdi, spike)
    self.r2o(tdi)
    self.r(tdi)
    self.o(tdi)
    return self.o.V.max(axis=1)


net = RSNN(100, 4, 2)

num_step = 200
num_sample = int(1e4)
freq = 5  # Hz
mask = bm.random.rand(num_sample, num_step, net.num_in)
x_data = bm.zeros((num_sample, num_step, net.num_in))
x_data[mask < freq * bm.get_dt() / 1000.] = 1.0
y_data = bm.asarray(bm.random.rand(num_sample) < 0.5, dtype=bm.get_dfloat())

trainer = bp.train.BPTT(net,
                        f_loss=bp.losses.cross_entropy_loss,
                        optimizer=bp.optim.Adam(lr=2e-3))
trainer.fit(train_data=(x_data, y_data), batch_size=100)

