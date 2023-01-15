# -*- coding: utf-8 -*-


import brainpy_datasets as bd
import jax.numpy as jnp
from tqdm import tqdm

import brainpy as bp
import brainpy.math as bm

traindata = bd.vision.MNIST(root='D:/data', split='train')
testdata = bd.vision.MNIST(root='D:/data', split='test')


def offline_train(num_hidden=2000, num_in=28, num_out=10):
  # training
  x_train = jnp.asarray(traindata.data / 255, dtype=bm.float_)
  x_train = x_train.reshape(-1, x_train.shape[-1])
  y_train = bm.one_hot(jnp.repeat(traindata.targets, x_train.shape[1]), 10, dtype=bm.float_)

  reservoir = bp.layers.Reservoir(
    num_in,
    num_hidden,
    Win_initializer=bp.init.Uniform(-0.6, 0.6),
    Wrec_initializer=bp.init.Normal(scale=0.1),
    in_connectivity=0.1,
    rec_connectivity=0.9,
    spectral_radius=1.3,
    leaky_rate=0.2,
    comp_type='dense',
    mode=bm.batching_mode
  )
  reservoir.reset_state(1)
  outs = bm.for_loop(bm.Partial(reservoir, {}), x_train)
  weight = bp.algorithms.RidgeRegression(alpha=1e-8)(y_train, outs)

  # predicting
  reservoir.reset_state(1)
  esn = bp.Sequential(
    reservoir,
    bp.layers.Dense(num_hidden,
                    num_out,
                    W_initializer=weight,
                    b_initializer=None,
                    mode=bm.training_mode)
  )

  preds = bm.for_loop(lambda x: jnp.argmax(esn({}, x), axis=-1),
                      x_train,
                      child_objs=esn)
  accuracy = jnp.mean(preds == jnp.repeat(traindata.targets, x_train.shape[1]))
  print(accuracy)


def force_online_train(num_hidden=2000, num_in=28, num_out=10, train_stage='final_step'):
  assert train_stage in ['final_step', 'all_steps']

  x_train = jnp.asarray(traindata.data / 255, dtype=bm.float_)
  x_test = jnp.asarray(testdata.data / 255, dtype=bm.float_)
  y_train = bm.one_hot(traindata.targets, 10, dtype=bm.float_)

  reservoir = bp.layers.Reservoir(
    num_in,
    num_hidden,
    Win_initializer=bp.init.Uniform(-0.6, 0.6),
    Wrec_initializer=bp.init.Normal(scale=1.3 / jnp.sqrt(num_hidden * 0.9)),
    in_connectivity=0.1,
    rec_connectivity=0.9,
    comp_type='dense',
    mode=bm.batching_mode
  )
  readout = bp.layers.Dense(num_hidden, num_out, b_initializer=None, mode=bm.training_mode)
  rls = bp.algorithms.RLS()
  rls.register_target(num_hidden)

  @bm.jit
  @bm.to_object(child_objs=(reservoir, readout, rls))
  def train_step(xs, y):
    reservoir.reset_state(xs.shape[0])
    if train_stage == 'final_step':
      for x in xs.transpose(1, 0, 2):
        o = reservoir(x)
      pred = readout(o)
      dw = rls(y, o, pred)
      readout.W += dw
    elif train_stage == 'all_steps':
      for x in xs.transpose(1, 0, 2):
        o = reservoir(x)
        pred = readout(o)
        dw = rls(y, o, pred)
        readout.W += dw
    else:
      raise ValueError

  @bm.jit
  @bm.to_object(child_objs=(reservoir, readout))
  def predict(xs):
    reservoir.reset_state(xs.shape[0])
    for x in xs.transpose(1, 0, 2):
      o = reservoir(x)
    y = readout(o)
    return jnp.argmax(y, axis=1)

  # training
  batch_size = 1
  for i in tqdm(range(0, x_train.shape[0], batch_size), desc='Training'):
    train_step(x_train[i: i + batch_size], y_train[i: i + batch_size])

  # verifying
  preds = []
  batch_size = 500
  for i in tqdm(range(0, x_train.shape[0], batch_size), desc='Verifying'):
    preds.append(predict(x_train[i: i + batch_size]))
  preds = jnp.concatenate(preds)
  acc = jnp.mean(preds == jnp.asarray(traindata.targets, dtype=bm.int_))
  print('Train accuracy', acc)

  # prediction
  preds = []
  for i in tqdm(range(0, x_test.shape[0], batch_size), desc='Predicting'):
    preds.append(predict(x_test[i: i + batch_size]))
  preds = jnp.concatenate(preds)
  acc = jnp.mean(preds == jnp.asarray(testdata.targets, dtype=bm.int_))
  print('Test accuracy', acc)


if __name__ == '__main__':
  # offline_train()
  force_online_train(num_hidden=2000)
