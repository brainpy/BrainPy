import brainpy_datasets as bd
from flax import linen as nn

import brainpy as bp
import brainpy.math as bm
from functools import partial

bm.set_environment(mode=bm.training_mode)


class CNN(nn.Module):
  """A CNN model implemented by using Flax."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    return x


class Network(bp.DynamicalSystem):
  def __init__(self):
    super(Network, self).__init__()
    self.cnn = bp.layers.FromFlax(CNN(), bm.ones([1, 4, 28, 1]))
    self.rnn = bp.layers.GRUCell(256, 100)
    self.linear = bp.layers.Dense(100, 10)

  def update(self, x):
    x = self.cnn(x)
    x = self.rnn(x)
    x = self.linear(x)
    return x


net = Network()
opt = bp.optim.Momentum(0.1)
data = bd.vision.MNIST(r'D:\data', download=True)
data.data = data.data.reshape(-1, 7, 4, 28, 1) / 255


def get_data(batch_size):
  key = bm.random.split_key()
  data.data = bm.random.permutation(data.data, key=key)
  data.targets = bm.random.permutation(data.targets, key=key)

  for i in range(0, len(data), batch_size):
    yield data.data[i: i + batch_size], data.targets[i: i + batch_size]


def loss_func(predictions, targets):
  logits = bm.max(predictions, axis=1)
  loss = bp.losses.cross_entropy_loss(logits, targets)
  accuracy = bm.mean(bm.argmax(logits, -1) == targets)
  return loss, {'accuracy': accuracy}


trainer = bp.BPTT(net, loss_fun=loss_func, optimizer=opt, loss_has_aux=True)
trainer.fit(partial(get_data, batch_size=256), num_epoch=10)

