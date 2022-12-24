# -*- coding: utf-8 -*-

import brainpy_datasets as bd

import brainpy as bp
import brainpy.math as bm


class FeedForwardModel(bp.DynamicalSystem):
  def __init__(self):
    super(FeedForwardModel, self).__init__()
    self.conv1 = bp.layers.Conv2D(1, 32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
    self.pool = bp.layers.MaxPool(2, 2, channel_axis=-1)
    self.conv2 = bp.layers.Conv2D(32, 64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
    self.fc1 = bp.layers.Dense(64 * 7 * 7, 1024)  # 两个池化，所以是7*7而不是14*14
    self.fc2 = bp.layers.Dense(1024, 512)
    self.fc3 = bp.layers.Dense(512, 10)

  def update(self, s, x):
    x = self.pool(s, bm.relu(self.conv1(s, x)))
    x = self.pool(s, bm.relu(self.conv2(s, x)))
    x = x.reshape(-1, 64 * 7 * 7)  # 将数据平整为一维的
    x = bm.relu(self.fc1(s, x))
    x = bm.relu(self.fc2(s, x))
    x = self.fc3(s, x)
    return x


# train dataset
train_dataset = bd.vision.MNIST(root=r'D:/data', split='train', download=True)
test_dataset = bd.vision.MNIST(root=r'D:/data', split='test', download=True)

num_batch = 128


def get_data(dataset):
  def generator():
    X = bm.expand_dims(bm.asarray(dataset.data/255, dtype=bm.float_), -1)
    Y = bm.asarray(dataset.targets, dtype=bm.int_)
    key = bm.random.DEFAULT.split_key()
    X = bm.random.permutation(X, key=key)
    Y = bm.random.permutation(Y, key=key)
    for i in range(0, X.shape[0], num_batch):
      yield X[i: i + num_batch], Y[i: i + num_batch]
  return generator


# model
with bm.environment(mode=bm.training_mode):
  model = FeedForwardModel()

# training
trainer = bp.train.BPFF(model,
                        loss_fun=bp.losses.cross_entropy_loss,
                        optimizer=bp.optim.Adam(lr=1e-3))
trainer.fit(get_data(train_dataset),
            get_data(test_dataset),
            num_epoch=2)
