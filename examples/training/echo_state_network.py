# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm


def esn(num_in=100, num_out=30):
  model = (
      bp.nn.Input(num_in)
      >>
      bp.nn.Reservoir(2000,
                      ff_initializer=bp.init.Uniform(-0.1, 0.1),
                      rec_initializer=bp.init.Normal(scale=0.1),
                      fb_initializer=bp.init.Uniform(-0.1, 0.1),
                      ff_connectivity=0.02,
                      fb_connectivity=0.02,
                      rec_connectivity=0.02,
                      name='l1',
                      conn_type='dense')
      >>
      bp.nn.LinearReadout(num_out, weight_initializer=bp.init.Normal(), name='l2')
  )
  model &= (model['l1'] << model['l2'])
  model.initialize(num_batch=1)

  # input-output
  print(model(bm.ones((1, num_in))))

  X = bm.random.random((1, 200, num_in))
  Y = bm.random.random((1, 200, num_out))

  # prediction
  runner = bp.nn.RNNRunner(model, monitors=['l1.state', 'l2.state'])
  outputs = runner.predict(X)
  print(runner.mon['l1.state'].shape)
  print(runner.mon['l2.state'].shape)
  print(bp.losses.mean_absolute_error(outputs, Y))
  print()

  # training
  trainer = bp.nn.RidgeTrainer(model)
  trainer.fit([X, Y])

  # prediction
  runner = bp.nn.RNNRunner(model, monitors=['l1.state', 'l2.state'], jit=True)
  outputs = runner.predict(X)
  print(runner.mon['l1.state'].shape)
  print(runner.mon['l2.state'].shape)
  print(bp.losses.mean_absolute_error(outputs, Y))
  print()

  outputs = trainer.predict(X)
  print(bp.losses.mean_absolute_error(outputs, Y))

  bp.base.clear_name_cache()


def train_esn_with_force(num_in=100, num_out=30):
  model = (
      bp.nn.Input(num_in)
      >>
      bp.nn.Reservoir(2000,
                      ff_initializer=bp.init.Uniform(-0.1, 0.1),
                      rec_initializer=bp.init.Normal(scale=0.1),
                      fb_initializer=bp.init.Uniform(-0.1, 0.1),
                      ff_connectivity=0.02,
                      fb_connectivity=0.02,
                      rec_connectivity=0.02,
                      name='l1',
                      conn_type='dense')
      >>
      bp.nn.LinearReadout(num_out, weight_initializer=bp.init.Normal(), name='l2')
  )
  model &= (model['l1'] << model['l2'])
  model.initialize(num_batch=1)

  # input-output
  print(model(bm.ones((1, num_in))))

  X = bm.random.random((1, 200, num_in))
  Y = bm.random.random((1, 200, num_out))

  # training
  trainer = bp.nn.ForceTrainer(model, alpha=0.1)
  trainer.fit([X, Y])

  # prediction
  runner = bp.nn.RNNRunner(model, monitors=['l1.state', 'l2.state'], jit=True)
  outputs = runner.predict(X)
  print(runner.mon['l1.state'].shape)
  print(runner.mon['l2.state'].shape)
  print(bp.losses.mean_absolute_error(outputs, Y))
  print()

  outputs = trainer.predict(X)
  print(bp.losses.mean_absolute_error(outputs, Y))

  bp.base.clear_name_cache()


def ngrc(num_in=10, num_out=30):
  bp.base.clear_name_cache()
  model = (bp.nn.Input(num_in)
           >> bp.nn.NVAR(delay=2, order=2, name='l1')
           >> bp.nn.Dense(num_out, weight_initializer=bp.init.Normal(0.1), trainable=True))
  model.initialize(num_batch=1)

  X = bm.random.random((1, 200, num_in))  # (num_batch, num_time, num_feature)
  Y = bm.random.random((1, 200, num_out))
  trainer = bp.nn.RidgeTrainer(model, beta=1e-6)
  outputs = trainer.predict(X)
  print(outputs.shape)
  print(bp.losses.mean_absolute_error(outputs, Y))
  trainer.fit([X, Y])
  outputs = trainer.predict(X)
  print(bp.losses.mean_absolute_error(outputs, Y))


def ngrc_bacth(num_in=10, num_out=30):
  bp.base.clear_name_cache()
  model = (
      bp.nn.Input(num_in)
      >>
      bp.nn.NVAR(delay=2, order=2, name='l1')
      >>
      bp.nn.Dense(num_out, weight_initializer=bp.init.Normal(0.1), trainable=True)
  )
  batch_size = 10
  model.initialize(num_batch=batch_size)

  X = bm.random.random((batch_size, 200, num_in))
  Y = bm.random.random((batch_size, 200, num_out))
  trainer = bp.nn.RidgeTrainer(model, beta=1e-6)
  outputs = trainer.predict(X)
  print(bp.losses.mean_absolute_error(outputs, Y))
  trainer.fit([X, Y])
  outputs = trainer.predict(X)
  print(bp.losses.mean_absolute_error(outputs, Y))


if __name__ == '__main__':
  train_esn_with_force(10, 30)
  print('ESN')
  esn(10, 30)
  print('NGRC')
  ngrc(10, 30)
  ngrc_bacth()
