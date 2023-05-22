import brainpy_datasets as bp_data
import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp
import brainpy.math as bm

# data
ds = bp_data.cognitive.RatePerceptualDecisionMaking(
  dt=20.,
  t_fixation=lambda: np.random.choice((50, 100, 200, 400)),
  t_stimulus=lambda: np.random.choice((100, 200, 400, 800)),
  num_trial=64 * 100
)
loader = bp_data.cognitive.TaskLoader(ds,
                                      max_seq_len=100,
                                      batch_size=64,
                                      data_first_axis='T')


# EI RNN model
class EI_RNN(bp.DynamicalSystem):
  def __init__(
      self, num_input, num_hidden, num_output, dt,
      e_ratio=0.8, sigma_rec=0., seed=None,
      w_ir=bp.init.KaimingUniform(scale=1.),
      w_rr=bp.init.KaimingUniform(scale=1.),
      w_ro=bp.init.KaimingUniform(scale=1.)
  ):
    super(EI_RNN, self).__init__()

    # parameters
    self.tau = 100
    self.num_input = num_input
    self.num_hidden = num_hidden
    self.num_output = num_output
    self.e_size = int(num_hidden * e_ratio)
    self.i_size = num_hidden - self.e_size
    self.alpha = dt / self.tau
    self.sigma_rec = (2 * self.alpha) ** 0.5 * sigma_rec  # Recurrent noise
    self.rng = bm.random.RandomState(seed=seed)

    # hidden mask
    mask = np.tile([1] * self.e_size + [-1] * self.i_size, (num_hidden, 1))
    np.fill_diagonal(mask, 0)
    self.mask = bm.asarray(mask, dtype=bm.float_)

    # input weight
    self.w_ir = bm.TrainVar(bp.init.parameter(w_ir, (num_input, num_hidden)))

    # recurrent weight
    bound = 1 / num_hidden ** 0.5
    self.w_rr = bm.TrainVar(bp.init.parameter(w_rr, (num_hidden, num_hidden)))
    self.w_rr[:, :self.e_size] /= (self.e_size / self.i_size)
    self.b_rr = bm.TrainVar(self.rng.uniform(-bound, bound, num_hidden))

    # readout weight
    bound = 1 / self.e_size ** 0.5
    self.w_ro = bm.TrainVar(bp.init.parameter(w_ro, (self.e_size, num_output)))
    self.b_ro = bm.TrainVar(self.rng.uniform(-bound, bound, num_output))

  def reset_state(self, batch_size):
    self.h = bm.Variable(bm.zeros((batch_size, self.num_hidden)), batch_axis=0)
    self.o = bm.Variable(bm.zeros((batch_size, self.num_output)), batch_axis=0)

  def cell(self, x, h):
    ins = x @ self.w_ir + h @ (bm.abs(self.w_rr) * self.mask) + self.b_rr
    state = h * (1 - self.alpha) + ins * self.alpha
    state += self.sigma_rec * self.rng.randn(self.num_hidden)
    return bm.relu(state)

  def readout(self, h):
    return h @ self.w_ro + self.b_ro

  @bp.not_pass_shared
  def update(self, x):
    self.h.value = self.cell(x, self.h)
    self.o.value = self.readout(self.h[:, :self.e_size])
    return self.h.value, self.o.value

  def predict(self, xs):
    self.h[:] = 0.
    return bm.for_loop(self.update, xs)

  def loss(self, xs, ys):
    hs, os = self.predict(xs)
    l = bp.losses.cross_entropy_loss(os.reshape((-1, os.shape[-1])), ys.flatten())
    acc = bm.mean(bm.argmax(os, axis=-1) == ys)
    return l, acc


# Instantiate the network and print information
hidden_size = 50
net = EI_RNN(num_input=len(ds.input_features),
             num_hidden=hidden_size,
             num_output=len(ds.output_features),
             dt=ds.dt,
             sigma_rec=0.15)


# Adam optimizer
opt = bp.optim.Adam(lr=0.001, train_vars=net.train_vars().unique())


# gradient function
grad_f = bm.grad(net.loss,
                 grad_vars=net.train_vars().unique(),
                 return_value=True,
                 has_aux=True)


# training function
@bm.jit
def train(xs, ys):
  grads, loss, acc = grad_f(xs, ys)
  opt.update(grads)
  return loss, acc


# training
for epoch_i in range(30):
  losses = []
  accs = []
  for x, y in loader:
    net.reset_state(x.shape[1])
    l, a = train(x, y)
    losses.append(l)
    accs.append(a)
  print(f'Epoch {epoch_i}, loss {np.mean(losses)}, acc {np.mean(accs)}')


# testing
ds.t_fixation = 500.  # set the fixed time duration for fixation and stimulus
ds.t_stimulus = 500.
x, y = zip(*[ds[i] for i in range(50)])  # get 50 trials
x = np.asarray(x).transpose(1, 0, 2)
y = np.asarray(y).transpose(1, 0)
net.reset_state(x.shape[1])
rnn_activity, action_pred = net.predict(x)
rnn_activity = bm.as_numpy(rnn_activity)
choice = np.argmax(bm.as_numpy(action_pred[-1]), axis=1)
correct = choice == y[-1]
print('Average performance', np.mean(correct))

# plot activity
trial = 0
plt.figure(figsize=(8, 6))
_ = plt.plot(rnn_activity[:, trial, :net.e_size], color='blue', label='Excitatory')
_ = plt.plot(rnn_activity[:, trial, net.e_size:], color='red', label='Inhibitory')
plt.xlabel('Time step')
plt.ylabel('Activity')
plt.show()


