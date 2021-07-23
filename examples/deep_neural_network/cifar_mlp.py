# -*- coding: utf-8 -*-


import random

import numpy as np
import tensorflow as tf

import brainpy as bp

# Data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.transpose(0, 3, 1, 2) / 255.0
X_test = X_test.transpose(0, 3, 1, 2) / 255.0

# Model
model = bp.dnn.MLP(layer_sizes=(100, 200, 300, 200))
opt = bp.dnn.Adam(target=model, lr=0.1, )


def loss(x, label):
  logit = model(x, config={'train': True})
  return bp.dnn.cross_entropy_sparse(logit, label).mean()


gv = bp.math.value_and_grad(loss, model.vars())


@bp.math.jit
@bp.math.function(nodes=(model, opt))
def train_op(x, y):
  g, v = gv(x, y)
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


# Training
for epoch in range(30):
  # Train
  loss = []
  sel = np.arange(len(X_train))
  np.random.shuffle(sel)
  for it in range(0, X_train.shape[0], 64):
    loss.append(train_op(augment(X_train[sel[it:it + 64]]),
                         Y_train[sel[it:it + 64]].flatten(),
                         4e-3 if epoch < 20 else 4e-4))

  # Eval
  test_predictions = [predict(x_batch).argmax(1)
                      for x_batch in X_test.reshape((50, -1) + X_test.shape[1:])]
  accuracy = np.array(test_predictions).flatten() == Y_test.flatten()
  print(f'Epoch {epoch + 1:4d}  Loss {np.mean(loss):.2f}  '
        f'Accuracy {100 * np.mean(accuracy):.2f}')
