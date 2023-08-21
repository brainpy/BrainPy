import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized
from absl.testing import absltest

dt = 0.04
num_step = int(1.0 / dt)
num_batch = 128


@bm.jit
def build_inputs_and_targets(mean=0.025, scale=0.01):
  sample = bm.random.normal(size=(num_batch, 1, 1))
  bias = mean * 2.0 * (sample - 0.5)
  samples = bm.random.normal(size=(num_batch, num_step, 1))
  noise_t = scale / dt ** 0.5 * samples
  inputs = bias + noise_t
  targets = bm.cumsum(inputs, axis=1)
  return inputs, targets


def train_data():
  for _ in range(100):
    yield build_inputs_and_targets()


class RNN(bp.DynamicalSystem):
  def __init__(self, num_in, num_hidden):
    super(RNN, self).__init__()
    self.rnn = bp.dnn.RNNCell(num_in, num_hidden, train_state=True)
    self.out = bp.dnn.Dense(num_hidden, 1)

  def update(self, x):
    return self.out(self.rnn(x))


with bm.training_environment():
  model = RNN(1, 100)


def loss(predictions, targets, l2_reg=2e-4):
  mse = bp.losses.mean_squared_error(predictions, targets)
  l2 = l2_reg * bp.losses.l2_norm(model.train_vars().unique().dict()) ** 2
  return mse + l2


class test_ModifyLr(parameterized.TestCase):
  @parameterized.product(
    LearningRate=[
      bp.optim.ExponentialDecayLR(lr=bm.Variable(bm.as_jax(0.025)), decay_steps=1, decay_rate=0.99975),
      bp.optim.InverseTimeDecayLR(lr=bm.Variable(bm.as_jax(0.025)), decay_steps=1, decay_rate=0.99975),
      bp.optim.PolynomialDecayLR(lr=bm.Variable(bm.as_jax(0.1)), decay_steps=1, final_lr=0.025),
      bp.optim.PiecewiseConstantLR(boundaries=(2, 2), values=(2, 2, 2))
    ]
  )
  def test_NewScheduler(self, LearningRate):
    opt = bp.optim.Adam(lr=LearningRate, eps=1e-1)
    trainer = bp.BPTT(model, loss_fun=loss, optimizer=opt)

    bm.clear_buffer_memory()

  def test_modifylr(self):
    Scheduler_lr = bp.optim.ExponentialDecayLR(lr=0.025, decay_steps=1, decay_rate=0.99975)

    opt1 = bp.optim.Adam(lr=Scheduler_lr, eps=1e-1)
    opt1.lr.lr = 0.01
    trainer1 = bp.BPTT(model, loss_fun=loss, optimizer=opt1)
    bm.clear_buffer_memory()

    opt2 = bp.optim.SGD(lr=Scheduler_lr)
    opt2.lr.set_value(0.01)
    trainer2 = bp.BPTT(model, loss_fun=loss, optimizer=opt2)
    bm.clear_buffer_memory()


if __name__ == '__main__':
  absltest.main()
