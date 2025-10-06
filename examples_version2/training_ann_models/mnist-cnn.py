# -*- coding: utf-8 -*-

import brainpy_datasets as bd
import jax.numpy as jnp

import brainpy.version2 as bp
import brainpy.version2.math as bm


class FeedForwardModel(bp.DynamicalSystem):
    def __init__(self):
        super(FeedForwardModel, self).__init__()
        self.conv1 = bp.dnn.Conv2d(1, 32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.pool = bp.dnn.MaxPool(2, 2, channel_axis=-1)
        self.conv2 = bp.dnn.Conv2d(32, 64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.fc1 = bp.dnn.Dense(64 * 7 * 7, 1024)
        self.fc2 = bp.dnn.Dense(1024, 512)
        self.fc3 = bp.dnn.Dense(512, 10)

    def update(self, x):
        x = self.pool(bm.relu(self.conv1(x)))
        x = self.pool(bm.relu(self.conv2(x)))
        x = x.reshape(-1, 64 * 7 * 7)
        x = bm.relu(self.fc1(x))
        x = bm.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# train dataset
train_dataset = bd.vision.MNIST(root=r'D:/data', split='train', download=True)
test_dataset = bd.vision.MNIST(root=r'D:/data', split='test', download=True)

num_batch = 128


def get_data(dataset):
    def generator():
        X = jnp.expand_dims(jnp.asarray(dataset.data / 255, dtype=bm.float_), -1)
        Y = jnp.asarray(dataset.targets, dtype=bm.int_)
        key = bm.random.DEFAULT.split_key()
        X = bm.random.permutation(X, key=key)
        Y = bm.random.permutation(Y, key=key)
        for i in range(0, X.shape[0], num_batch):
            yield X[i: i + num_batch], Y[i: i + num_batch]

    return generator


# model
with bm.training_environment():
    model = FeedForwardModel()

# training
trainer = bp.BPFF(model,
                  loss_fun=bp.losses.cross_entropy_loss,
                  optimizer=bp.optim.Adam(lr=1e-3))
trainer.fit(get_data(train_dataset),
            get_data(test_dataset),
            num_epoch=2)
