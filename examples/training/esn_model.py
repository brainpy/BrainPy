# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm


model = (
  bp.nn.Input(3)
  >>
  bp.nn.Reservoir(100,
                  init_ff=bp.init.Uniform(-0.1, 0.1),
                  init_rec=bp.init.Normal(scale=0.1),
                  init_fb=bp.init.Uniform(-0.1, 0.1),
                  name='l1')
  >>
  bp.nn.LinearReadout(3, init_weight=bp.init.Normal(), name='l2')
)
model &= (model['l1'] << model['l2'])

# input-output
print(model(bm.ones(3)))


X = bm.random.random((200, 3))
Y = bm.random.random((200, 3))

# prediction
runner = bp.nn.RNNRunner(model, monitors=['l1.state', 'l2.state'])
outputs = runner.predict(X)
print(runner.mon['l1.state'].shape)
print(runner.mon['l2.state'].shape)
print(bp.losses.mean_absolute_error(outputs, Y))
print()

# training
trainer = bp.nn.RidgeTrainer(model)
trainer.fit(X, Y)
outputs = trainer.predict(X)
print(bp.losses.mean_absolute_error(outputs, Y))

