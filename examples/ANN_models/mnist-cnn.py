# -*- coding: utf-8 -*-

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
train_dataset = bp.datasets.MNIST(root='./data', train=True, download=True)
x_train = bm.array(train_dataset.data, dtype=bm.dftype())
x_train = x_train.reshape(x_train.shape + (1,)) / 255
y_train = bm.array(train_dataset.targets, dtype=bm.ditype())

# model
model = FeedForwardModel()

# training
trainer = bp.train.BPFF(model,
                        loss_fun=bp.losses.cross_entropy_loss,
                        optimizer=bp.optim.Adam(lr=1e-3),
                        shuffle_data=True)
trainer.fit([x_train, y_train], num_epoch=2, batch_size=64)

# test dataset
test_dataset = bp.datasets.MNIST(root='./data', train=False, download=True)
x_test = bm.array(test_dataset.data, dtype=bm.dftype())
x_test = x_test.reshape(x_test.shape + (1,)) / 255
y_test = bm.array(test_dataset.targets, dtype=bm.ditype())

# testing
y_predicts = []
for i in range(0, x_test.shape[0], 100):
  y_predicts.append(bm.argmax(trainer.predict(x_test[i: i + 100]), axis=1))
acc = bm.mean(bm.concatenate(y_predicts) == y_test)  # compare to labels
print("Test Accuracy %.5f" % acc)

