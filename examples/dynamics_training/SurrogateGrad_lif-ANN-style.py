# -*- coding: utf-8 -*-


"""
Reproduce the results of the``spytorch`` tutorial 1:

- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial1.ipynb

"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import brainpy as bp
import brainpy.math as bm


class SNN(bp.dyn.Network):
  def __init__(self, num_in, num_rec, num_out):
    super(SNN, self).__init__()

    # parameters
    self.num_in = num_in
    self.num_rec = num_rec
    self.num_out = num_out

    # neuron groups
    self.i = bp.neurons.InputGroup(num_in, mode=bp.modes.training)
    self.r = bp.neurons.LIF(num_rec, tau=10, V_reset=0, V_rest=0, V_th=1., mode=bp.modes.training)
    self.o = bp.neurons.LeakyIntegrator(num_out, tau=5, mode=bp.modes.training)

    # synapse: i->r
    self.i2r = bp.synapses.Exponential(self.i, self.r, bp.conn.All2All(), tau=10.,
                                       output=bp.synouts.CUBA(target_var=None),
                                       g_max=bp.init.KaimingNormal(scale=20.),
                                       mode=bp.modes.training)
    # synapse: r->o
    self.r2o = bp.synapses.Exponential(self.r, self.o, bp.conn.All2All(), tau=10.,
                                       output=bp.synouts.CUBA(target_var=None),
                                       g_max=bp.init.KaimingNormal(scale=20.),
                                       mode=bp.modes.training)

    # whole model
    self.model = bp.Sequential(self.i, self.i2r, self.r, self.r2o, self.o)

  def update(self, tdi, spike):
    self.model(tdi, spike)
    return self.o.V.value


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
  gs = GridSpec(*dim)
  mem = 1. * mem
  if spk is not None:
    mem[spk > 0.0] = spike_height
  mem = bm.as_numpy(mem)
  for i in range(np.prod(dim)):
    if i == 0:
      a0 = ax = plt.subplot(gs[i])
    else:
      ax = plt.subplot(gs[i], sharey=a0)
    ax.plot(mem[i])
  plt.tight_layout()
  plt.show()


def print_classification_accuracy(output, target):
  """ Dirty little helper function to compute classification accuracy. """
  m = bm.max(output, axis=1)  # max over time
  am = bm.argmax(m, axis=1)  # argmax over output units
  acc = bm.mean(target == am)  # compare to labels
  print("Accuracy %.3f" % acc)


net = SNN(100, 4, 2)

num_step = 2000
num_sample = 256
freq = 5  # Hz
mask = bm.random.rand(num_sample, num_step, net.num_in)
x_data = bm.zeros((num_sample, num_step, net.num_in))
x_data[mask < freq * bm.get_dt() / 1000.] = 1.0
y_data = bm.asarray(bm.random.rand(num_sample) < 0.5, dtype=bm.dftype())
rng = bm.random.RandomState()

# Before training
runner = bp.dyn.DSRunner(net, monitors={'r.spike': net.r.spike, 'r.membrane': net.r.V})
out = runner.run(inputs=x_data, inputs_are_batching=True, reset_state=True)
plot_voltage_traces(runner.mon.get('r.membrane'), runner.mon.get('r.spike'))
plot_voltage_traces(out)
print_classification_accuracy(out, y_data)


@bm.to_object(child_objs=net, dyn_vars=rng)  # add nodes and vars used in this function
def loss():
  key = rng.split_key()
  X = bm.random.permutation(x_data, key=key)
  Y = bm.random.permutation(y_data, key=key)
  looper = bp.dyn.DSRunner(net, numpy_mon_after_run=False, progress_bar=False)
  predictions = looper.run(inputs=X, inputs_are_batching=True, reset_state=True)
  predictions = bm.max(predictions, axis=1)
  return bp.losses.cross_entropy_loss(predictions, Y)


grad = bm.grad(loss, grad_vars=loss.train_vars().unique(), return_value=True)
optimizer = bp.optim.Adam(lr=2e-3, train_vars=net.train_vars().unique())


@bm.to_object(child_objs=(grad, optimizer))  # add nodes and vars used in this function
def train(_):
  grads, l = grad()
  optimizer.update(grads)
  return l


# train the network
net.reset_state(num_sample)
train_losses = []
for i in range(0, 3000, 100):
  t0 = time.time()
  ls = bm.for_loop(train, operands=bm.arange(i, i + 100, 1))
  print(f'Train {i + 100} epoch, loss = {bm.mean(ls):.4f}, used time {time.time() - t0:.4f} s')
  train_losses.append(ls)

# visualize the training losses
plt.plot(bm.as_numpy(bm.concatenate(train_losses)))
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.show()

# predict the output according to the input data
runner = bp.dyn.DSRunner(net, monitors={'r.spike': net.r.spike, 'r.membrane': net.r.V})
out = runner.run(inputs=x_data, inputs_are_batching=True, reset_state=True)
plot_voltage_traces(runner.mon.get('r.membrane'), runner.mon.get('r.spike'))
plot_voltage_traces(out)
print_classification_accuracy(out, y_data)
