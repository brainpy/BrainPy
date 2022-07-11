# -*- coding: utf-8 -*-

"""Implementation of the paper:

- Gauthier, D.J., Bollt, E., Griffith, A. et al. Next generation reservoir
  computing. Nat Commun 12, 5564 (2021). https://doi.org/10.1038/s41467-021-25801-2

The main task is forecasting the double-scroll system.
"""


import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp
import brainpy.math as bm

bm.enable_x64()


def get_subset(data, start, end):
  res = {'x': data['x'][start: end],
         'y': data['y'][start: end],
         'z': data['z'][start: end]}
  res = bm.hstack([res['x'], res['y'], res['z']])
  return res.reshape((1,) + res.shape)


def plot_weights(Wout, coefs, bias=None):
  Wout = np.asarray(Wout)
  if bias is not None:
    bias = np.asarray(bias)
    Wout = np.concatenate([bias.reshape((1, 3)), Wout], axis=0)
    coefs.insert(0, 'bias')
  x_Wout, y_Wout, z_Wout = Wout[:, 0], Wout[:, 1], Wout[:, 2]

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(131)
  ax.grid(axis="y")
  ax.set_xlabel("$[W_{out}]_x$")
  ax.set_ylabel("Features")
  ax.set_yticks(np.arange(len(coefs)))
  ax.set_yticklabels(coefs)
  ax.barh(np.arange(x_Wout.size), x_Wout)

  ax1 = fig.add_subplot(132)
  ax1.grid(axis="y")
  ax1.set_yticks(np.arange(len(coefs)))
  ax1.set_xlabel("$[W_{out}]_y$")
  ax1.barh(np.arange(y_Wout.size), y_Wout)

  ax2 = fig.add_subplot(133)
  ax2.grid(axis="y")
  ax2.set_yticks(np.arange(len(coefs)))
  ax2.set_xlabel("$[W_{out}]_z$")
  ax2.barh(np.arange(z_Wout.size), z_Wout)

  plt.show()


def plot_double_scroll(ground_truth, predictions):
  fig = plt.figure(figsize=(15, 10))
  ax = fig.add_subplot(121, projection='3d')
  ax.set_title("Generated attractor")
  ax.set_xlabel("$x$")
  ax.set_ylabel("$y$")
  ax.set_zlabel("$z$")
  ax.grid(False)
  ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2])

  ax2 = fig.add_subplot(122, projection='3d')
  ax2.set_title("Real attractor")
  ax2.grid(False)
  ax2.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2])
  plt.show()


dt = 0.02
t_warmup = 10.  # ms
t_train = 100.  # ms
t_test = 800.  # ms
num_warmup = int(t_warmup / dt)  # warm up NVAR
num_train = int(t_train / dt)
num_test = int(t_test / dt)

# Datasets #
# -------- #
data_series = bp.datasets.double_scroll_series(t_warmup + t_train + t_test, dt=dt)

X_warmup = get_subset(data_series, 0, num_warmup - 1)
Y_warmup = get_subset(data_series, 1, num_warmup)
X_train = get_subset(data_series, num_warmup - 1, num_warmup + num_train - 1)
# Target: Lorenz[t] - Lorenz[t - 1]
dX_train = get_subset(data_series, num_warmup, num_warmup + num_train) - X_train
X_test = get_subset(data_series,
                    num_warmup + num_train - 1,
                    num_warmup + num_train + num_test - 1)
Y_test = get_subset(data_series,
                    num_warmup + num_train,
                    num_warmup + num_train + num_test)

# Model #
# ----- #


class NGRC(bp.dyn.DynamicalSystem):
  def __init__(self, num_in):
    super(NGRC, self).__init__()
    self.r = bp.layers.NVAR(num_in, delay=2, order=3, mode=bp.modes.batching)
    self.di = bp.layers.Dense(self.r.num_out, num_in, mode=bp.modes.training)

  def update(self, shared, x):
    di = self.di(shared, self.r(shared, x))
    return x + di


model = NGRC(3)

# Training #
# -------- #

# warm-up
trainer = bp.train.RidgeTrainer(model, alpha=1e-5, jit=True)
outputs = trainer.predict(X_warmup)
print('Warmup NMS: ', bp.losses.mean_squared_error(outputs, Y_warmup))

# training
trainer.fit([X_train, {'di': dX_train}])
plot_weights(model.di.W, model.r.get_feature_names(for_plot=True), model.di.b)

# prediction
shared = dict()
model_jit = bm.jit(model)
outputs = [model_jit(shared, X_test[:, 0])]
for i in range(1, X_test.shape[1]):
  outputs.append(model_jit(shared, outputs[i - 1]))
outputs = bm.asarray(outputs).squeeze()
print('Prediction NMS: ', bp.losses.mean_squared_error(outputs, Y_test))
plot_double_scroll(bm.as_numpy(Y_test).squeeze(), bm.as_numpy(outputs))
