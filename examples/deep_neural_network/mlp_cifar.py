# -*- coding: utf-8 -*-


import random
import numpy as np
import tensorflow as tf

import brainpy as bp
bp.math.use_backend('jax')


# Data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
num_train, num_test = X_train.shape[0], X_test.shape[0]
num_dim = bp.tools.size2num(X_train.shape[1:])
X_train = X_train.reshape((num_train, num_dim)) / 255.0
X_test = X_test.reshape((num_test, num_dim)) / 255.0
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()


# Model
model = bp.dnn.MLP(layer_sizes=(num_dim, 256, 512, 256, 10))
opt = bp.dnn.Adam(lr=0.001, train_vars=model.train_vars())


# loss

def loss_func(x, label):
  logit = model(x, config={'train': True})
  return bp.dnn.cross_entropy(logit, label).mean()


vg = bp.math.value_and_grad(loss_func, model.train_vars())


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


def augment(x):
  if random.random() < .5:
    x = x[:, :, :, ::-1]  # Flip the batch images about the horizontal axis
  # Pixel-shift all images in the batch by up to 4 pixels in any direction.
  x_pad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], 'reflect')
  rx, ry = np.random.randint(0, 8), np.random.randint(0, 8)
  x = x_pad[:, :, rx:rx + 32, ry:ry + 32]
  return x


num_batch = 128

# Training
for epoch in range(30):
  # Train
  loss = []
  sel = np.arange(len(X_train))
  np.random.shuffle(sel)
  for it in range(0, X_train.shape[0], num_batch):
    l = train_op(X_train[sel[it:it + num_batch]], Y_train[sel[it:it + num_batch]])
    loss.append(l)

  # Eval
  test_predictions = [predict(x_batch).argmax(1) for x_batch in X_test.reshape((50, -1, num_dim))]
  accuracy = np.array(test_predictions).flatten() == Y_test.flatten()
  print(f'Epoch {epoch + 1:4d}  '
        f'Loss {np.mean(loss):.2f}  '
        f'Accuracy {100 * np.mean(accuracy):.2f}')
