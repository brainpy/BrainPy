# -*- coding: utf-8 -*-

"""Implementation of the paper:

- Gauthier, D.J., Bollt, E., Griffith, A. et al. Next generation reservoir
  computing. Nat Commun 12, 5564 (2021). https://doi.org/10.1038/s41467-021-25801-2

The main task is forecasting the Lorenz63 strange attractor.
"""

import brainpy_datasets as bp_data
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp
import brainpy as bp
import brainpy.math as bm


bm.set_environment(bm.batching_mode, x64=True)


def get_subset(data, start, end):
  res = {'x': data.xs[start: end],
         'y': data.ys[start: end],
         'z': data.zs[start: end]}
  X = jnp.hstack([res['x'], res['y']])
  X = X.reshape((1,) + X.shape)
  Y = res['z']
  Y = Y.reshape((1,) + Y.shape)
  return X, Y


def plot_lorenz(x, y, true_z, predict_z, linewidth=.8):
  fig1 = plt.figure()
  fig1.set_figheight(8)
  fig1.set_figwidth(12)

  t_all = t_warmup + t_train + t_test
  ts = np.arange(0, t_all, dt)

  h = 240
  w = 2

  # top left of grid is 0,0
  axs1 = plt.subplot2grid(shape=(h, w), loc=(0, 0), colspan=2, rowspan=30)
  axs2 = plt.subplot2grid(shape=(h, w), loc=(36, 0), colspan=2, rowspan=30)
  axs3 = plt.subplot2grid(shape=(h, w), loc=(72, 0), colspan=2, rowspan=30)
  axs4 = plt.subplot2grid(shape=(h, w), loc=(132, 0), colspan=2, rowspan=30)
  axs5 = plt.subplot2grid(shape=(h, w), loc=(168, 0), colspan=2, rowspan=30)
  axs6 = plt.subplot2grid(shape=(h, w), loc=(204, 0), colspan=2, rowspan=30)

  # training phase x
  axs1.set_title('training phase')
  axs1.plot(ts[num_warmup:num_warmup + num_train],
            x[num_warmup:num_warmup + num_train],
            color='b', linewidth=linewidth)
  axs1.set_ylabel('x')
  axs1.axes.xaxis.set_ticklabels([])
  axs1.axes.set_xbound(t_warmup - .08, t_warmup + t_train + .05)
  axs1.axes.set_ybound(-21., 21.)
  axs1.text(-.14, .9, 'a)', ha='left', va='bottom', transform=axs1.transAxes)

  # training phase y
  axs2.plot(ts[num_warmup:num_warmup + num_train],
            y[num_warmup:num_warmup + num_train],
            color='b', linewidth=linewidth)
  axs2.set_ylabel('y')
  axs2.axes.xaxis.set_ticklabels([])
  axs2.axes.set_xbound(t_warmup - .08, t_warmup + t_train + .05)
  axs2.axes.set_ybound(-26., 26.)
  axs2.text(-.14, .9, 'b)', ha='left', va='bottom', transform=axs2.transAxes)

  # training phase z
  axs3.plot(ts[num_warmup:num_warmup + num_train],
            true_z[num_warmup:num_warmup + num_train],
            color='b', linewidth=linewidth)
  axs3.plot(ts[num_warmup:num_warmup + num_train],
            predict_z[num_warmup:num_warmup + num_train],
            color='r', linewidth=linewidth)
  axs3.set_ylabel('z')
  axs3.set_xlabel('time')
  axs3.axes.set_xbound(t_warmup - .08, t_warmup + t_train + .05)
  axs3.axes.set_ybound(3., 48.)
  axs3.text(-.14, .9, 'c)', ha='left', va='bottom', transform=axs3.transAxes)

  # testing phase x
  axs4.set_title('testing phase')
  axs4.plot(ts[num_warmup + num_train:num_warmup + num_train + num_test],
            x[num_warmup + num_train:num_warmup + num_train + num_test],
            color='b', linewidth=linewidth)
  axs4.set_ylabel('x')
  axs4.axes.xaxis.set_ticklabels([])
  axs4.axes.set_ybound(-21., 21.)
  axs4.axes.set_xbound(t_warmup + t_train - .5, t_all + .5)
  axs4.text(-.14, .9, 'd)', ha='left', va='bottom', transform=axs4.transAxes)

  # testing phase y
  axs5.plot(ts[num_warmup + num_train:num_warmup + num_train + num_test],
            y[num_warmup + num_train:num_warmup + num_train + num_test],
            color='b', linewidth=linewidth)
  axs5.set_ylabel('y')
  axs5.axes.xaxis.set_ticklabels([])
  axs5.axes.set_ybound(-26., 26.)
  axs5.axes.set_xbound(t_warmup + t_train - .5, t_all + .5)
  axs5.text(-.14, .9, 'e)', ha='left', va='bottom', transform=axs5.transAxes)

  # testing phose z
  axs6.plot(ts[num_warmup + num_train:num_warmup + num_train + num_test],
            true_z[num_warmup + num_train:num_warmup + num_train + num_test],
            color='b', linewidth=linewidth)
  axs6.plot(ts[num_warmup + num_train:num_warmup + num_train + num_test],
            predict_z[num_warmup + num_train:num_warmup + num_train + num_test],
            color='r', linewidth=linewidth)
  axs6.set_ylabel('z')
  axs6.set_xlabel('time')
  axs6.axes.set_ybound(3., 48.)
  axs6.axes.set_xbound(t_warmup + t_train - .5, t_all + .5)
  axs6.text(-.14, .9, 'f)', ha='left', va='bottom', transform=axs6.transAxes)

  plt.show()


dt = 0.02
t_warmup = 10.  # ms
t_train = 20.  # ms
t_test = 50.  # ms
num_warmup = int(t_warmup / dt)  # warm up NVAR
num_train = int(t_train / dt)
num_test = int(t_test / dt)

# Datasets #
# -------- #
lorenz_series = bp_data.chaos.LorenzEq(t_warmup + t_train + t_test,
                                       dt=dt,
                                       inits={'x': 17.67715816276679,
                                              'y': 12.931379185960404,
                                              'z': 43.91404334248268})

X_warmup, Y_warmup = get_subset(lorenz_series, 0, num_warmup)
X_train, Y_train = get_subset(lorenz_series, num_warmup, num_warmup + num_train)
X_test, Y_test = get_subset(lorenz_series, 0, num_warmup + num_train + num_test)


# Model #
# ----- #

class NGRC(bp.DynamicalSystem):
  def __init__(self, num_in):
    super(NGRC, self).__init__()
    self.r = bp.layers.NVAR(num_in, delay=4, order=2, stride=5)
    self.o = bp.layers.Dense(self.r.num_out, 1, mode=bm.training_mode)

  def update(self, sha, x):
    return self.o(sha, self.r(sha, x))


model = NGRC(2)

# Training #
# -------- #

trainer = bp.RidgeTrainer(model, alpha=0.05)

# warm-up
outputs = trainer.predict(X_warmup)
print('Warmup NMS: ', bp.losses.mean_squared_error(outputs, Y_warmup))

# training
trainer.fit([X_train, Y_train])

# prediction
outputs = trainer.predict(X_test, reset_state=True)
print('Prediction NMS: ', bp.losses.mean_squared_error(outputs, Y_test))

plot_lorenz(x=bm.as_numpy(lorenz_series.xs).flatten(),
            y=bm.as_numpy(lorenz_series.ys).flatten(),
            true_z=bm.as_numpy(lorenz_series.zs).flatten(),
            predict_z=bm.as_numpy(outputs).flatten())
