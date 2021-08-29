# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf

import brainpy as bp

bp.math.use_backend('jax')

# Data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
num_train, num_test = X_train.shape[0], X_test.shape[0]
num_dim = bp.tools.size2num(X_train.shape[1:])
X_train = X_train.reshape((num_train, num_dim)) / 255.0
X_test = X_test.reshape((num_test, num_dim)) / 255.0
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

# Model
model = bp.dnn.MLP(layer_sizes=(num_dim, 256, 256, 10))
opt = bp.dnn.Adam(lr=0.0001, train_vars=model.train_vars().unique())


# loss

def loss_func(x, label):
  logit = model(x, config={'train': True})
  return bp.dnn.cross_entropy_loss(logit, label).mean()


vg = bp.math.value_and_grad(loss_func, model.vars())


# functions

@bp.math.jit
@bp.math.function(nodes=(model, opt))
def train_op(x, y):
  v, g = vg(x, y)
  opt(grads=g)
  return v


@bp.math.jit
@bp.math.function(nodes=model)
def predict(x):
  logit = model(x, config={'train': False})
  return bp.dnn.softmax(logit)


num_batch = 128

# Training
for epoch in range(30):
  # Train
  loss = []
  sel = np.arange(len(X_train))
  np.random.shuffle(sel)
  for it in range(0, X_train.shape[0], num_batch):
    l = train_op(X_train[sel[it:it + num_batch]], Y_train[sel[it:it + num_batch]])
    print(model.children_modules.keys())
    layer = model.children_modules['MLP0_l1']
    w1 = layer.w
    w2 = train_op.jit_vars['MLP0_l1.w']
    print(layer.w)
    loss.append(l)

  # Eval
  test_predictions = [predict(x_batch).argmax(1) for x_batch in X_test.reshape((50, -1, num_dim))]
  accuracy = np.array(test_predictions).flatten() == Y_test.flatten()
  print(f'Epoch {epoch + 1:4d}  '
        f'Loss {np.mean(loss):.3f}  '
        f'Accuracy {100 * np.mean(accuracy):.3f}')
