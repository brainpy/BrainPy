# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm


def esn(num_in=100, num_out=30):
  model = (
      bp.nn.Input(num_in)
      >>
      bp.nn.Reservoir(2000,
                      init_ff=bp.init.Uniform(-0.1, 0.1),
                      init_rec=bp.init.Normal(scale=0.1),
                      init_fb=bp.init.Uniform(-0.1, 0.1),
                      ff_connectivity=0.02,
                      fb_connectivity=0.02,
                      rec_connectivity=0.02,
                      name='l1',
                      conn_type='dense')
      >>
      bp.nn.LinearReadout(num_out, init_weight=bp.init.Normal(), name='l2')
  )
  model &= (model['l1'] << model['l2'])

  # input-output
  # print(model(bm.ones(num_in)))

  X = bm.random.random((200, num_in))
  Y = bm.random.random((200, num_out))

  # # prediction
  # runner = bp.nn.RNNRunner(model, monitors=['l1.state', 'l2.state'])
  # outputs = runner.predict(X)
  # print(runner.mon['l1.state'].shape)
  # print(runner.mon['l2.state'].shape)
  # print(bp.losses.mean_absolute_error(outputs, Y))
  # print()

  # training
  trainer = bp.nn.RidgeTrainer(model)
  trainer.fit(X, Y)

  # # prediction
  # runner = bp.nn.RNNRunner(model, monitors=['l1.state', 'l2.state'], jit=True)
  # outputs = runner.predict(X)
  # print(runner.mon['l1.state'].shape)
  # print(runner.mon['l2.state'].shape)
  # print(bp.losses.mean_absolute_error(outputs, Y))
  # print()

  outputs = trainer.predict(X)
  print(bp.losses.mean_absolute_error(outputs, Y))


def ngrc(num_in=10, num_out=30):
  bp.base.clear_name_cache()
  model = (bp.nn.Input(num_in)
           >> bp.nn.NVAR(delay=2, order=2, name='l1')
           >> bp.nn.Dense(num_out, init_weight=bp.init.Normal(0.1), trainable=True))

  X = bm.random.random((200, num_in))
  Y = bm.random.random((200, num_out))
  trainer = bp.nn.RidgeTrainer(model, beta=1e-6)
  outputs = trainer.predict(X)
  # print()
  # print(trainer.mon['l1.output'].shape)
  print(bp.losses.mean_absolute_error(outputs, Y))
  trainer.fit(X, Y)
  outputs = trainer.predict(X)
  print(bp.losses.mean_absolute_error(outputs, Y))


def ngrc_bacth(num_in=10, num_out=30):
  bp.base.clear_name_cache()
  model = (bp.nn.NVAR(delay=2, order=2, name='l1')
           >> bp.nn.Dense(num_out, init_weight=bp.init.Normal(0.1), trainable=True))

  batch_size = 10

  X = bm.random.random((200, batch_size, num_in))
  Y = bm.random.random((200, batch_size, num_out))
  trainer = bp.nn.RidgeTrainer(model, beta=1e-6, jit=False)
  outputs = trainer.predict(X)
  # print()
  # print(trainer.mon['l1.output'].shape)
  print(bp.losses.mean_absolute_error(outputs, Y))
  trainer.fit(X, Y)
  outputs = trainer.predict(X)
  print(bp.losses.mean_absolute_error(outputs, Y))


if __name__ == '__main__':
  # print('ESN')
  # esn(10, 30)
  # print('NGRC')
  ngrc(10, 30)
  # ngrc_bacth()
