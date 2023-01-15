# -*- coding: utf-8 -*-

from functools import partial

import matplotlib.pyplot as plt

import jax.numpy as jnp
import brainpy as bp
import brainpy.math as bm

dt = 0.04
num_step = int(1.0 / dt)
num_batch = 128


@partial(bm.jit, static_argnames=['batch_size'])
@bm.to_object(dyn_vars=bm.random.DEFAULT)
def build_inputs_and_targets(mean=0.025, scale=0.01, batch_size=10):
  # Create the white noise input
  sample = bm.random.normal(size=(batch_size, 1, 1))
  bias = mean * 2.0 * (sample - 0.5)
  samples = bm.random.normal(size=(batch_size, num_step, 1))
  noise_t = scale / dt ** 0.5 * samples
  inputs = bias + noise_t
  targets = jnp.cumsum(inputs, axis=1)
  return inputs, targets


def train_data():
  for _ in range(100):
    yield build_inputs_and_targets(batch_size=num_batch)


class RNN(bp.DynamicalSystem):
  def __init__(self, num_in, num_hidden):
    super(RNN, self).__init__()
    self.rnn = bp.layers.RNNCell(num_in, num_hidden, train_state=True)
    self.out = bp.layers.Dense(num_hidden, 1)

  def update(self, sha, x):
    return self.out(sha, self.rnn(sha, x))


with bm.training_environment():
  model = RNN(1, 100)


# define loss function
def loss(predictions, targets, l2_reg=2e-4):
  mse = bp.losses.mean_squared_error(predictions, targets)
  l2 = l2_reg * bp.losses.l2_norm(model.train_vars().unique().dict()) ** 2
  return mse + l2


# define optimizer
lr = bp.optim.ExponentialDecay(lr=0.025, decay_steps=1, decay_rate=0.99975)
opt = bp.optim.Adam(lr=lr, eps=1e-1)

# create a trainer
trainer = bp.BPTT(model, loss_fun=loss, optimizer=opt)
trainer.fit(train_data,
            num_epoch=30,
            num_report=200)

plt.plot(bm.as_numpy(trainer.get_hist_metric()))
plt.show()

model.reset_state(1)
x, y = build_inputs_and_targets(batch_size=1)
predicts = trainer.predict(x)

plt.figure(figsize=(8, 2))
plt.plot(bm.as_numpy(y[0]).flatten(), label='Ground Truth')
plt.plot(bm.as_numpy(predicts[0]).flatten(), label='Prediction')
plt.legend()
plt.show()
