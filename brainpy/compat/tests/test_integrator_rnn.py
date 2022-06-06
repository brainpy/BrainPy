# -*- coding: utf-8 -*-

from functools import partial

import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm


block = False
dt = 0.04
num_step = int(1.0 / dt)
num_batch = 128


@partial(bm.jit,
         dyn_vars=bp.TensorCollector({'a': bm.random.DEFAULT}),
         static_argnames=['batch_size'])
def build_inputs_and_targets(mean=0.025, scale=0.01, batch_size=10):
  # Create the white noise input
  sample = bm.random.normal(size=(batch_size, 1, 1))
  bias = mean * 2.0 * (sample - 0.5)
  samples = bm.random.normal(size=(batch_size, num_step, 1))
  noise_t = scale / dt ** 0.5 * samples
  inputs = bias + noise_t
  targets = bm.cumsum(inputs, axis=1)
  return inputs, targets


def train_data():
  for _ in range(10):
    yield build_inputs_and_targets(batch_size=num_batch)


def test_rnn_training():
  model = (
      bp.nn.Input(1)
      >>
      bp.nn.VanillaRNN(100, state_trainable=True)
      >>
      bp.nn.Dense(1)
  )
  model.initialize(num_batch=num_batch)


  # define loss function
  def loss(predictions, targets, l2_reg=2e-4):
    mse = bp.losses.mean_squared_error(predictions, targets)
    l2 = l2_reg * bp.losses.l2_norm(model.train_vars().unique().dict()) ** 2
    return mse + l2


  # define optimizer
  lr = bp.optim.ExponentialDecay(lr=0.025, decay_steps=1, decay_rate=0.99975)
  opt = bp.optim.Adam(lr=lr, eps=1e-1)

  # create a trainer
  trainer = bp.nn.BPTT(model,
                       loss=loss,
                       optimizer=opt,
                       max_grad_norm=5.0)
  trainer.fit(train_data,
              num_batch=num_batch,
              num_train=5,
              num_report=10)

  plt.plot(trainer.train_losses.numpy())
  plt.show(block=block)

  model.initialize(1)
  x, y = build_inputs_and_targets(batch_size=1)
  predicts = trainer.predict(x)

  plt.figure(figsize=(8, 2))
  plt.plot(bm.as_numpy(y[0]).flatten(), label='Ground Truth')
  plt.plot(bm.as_numpy(predicts[0]).flatten(), label='Prediction')
  plt.legend()
  plt.show(block=block)
  plt.close()


