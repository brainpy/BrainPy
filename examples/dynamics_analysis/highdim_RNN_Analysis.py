"""
Implementation of the paper:

- Yang G R, Wang X J. Artificial neural networks for neuroscientists:
  A primer[J]. Neuron, 2020, 107(6): 1048-1070.
"""

import brainpy as bp
import brainpy.math as bm

bp.math.set_platform('cpu')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# In this tutorial, we will use supervised learning to train a recurrent
# neural network on a simple perceptual decision making task, and analyze
# the trained network using dynamical system analysis.

# Defining a cognitive task
# ----
# We will import the task from the neurogym library.
# Please install neurogym:
# https://github.com/neurogym/neurogym

import neurogym as ngym

# Environment
task = 'PerceptualDecisionMaking-v0'
kwargs = {'dt': 100}
seq_len = 100

# Make supervised dataset
dataset = ngym.Dataset(task,
                       env_kwargs=kwargs,
                       batch_size=16,
                       seq_len=seq_len)

# A sample environment from dataset
env = dataset.env


# Define a vanilla continuous-time recurrent network
class RNNNet(bp.DynamicalSystem):
  def __init__(
      self,
      num_input, num_hidden, num_output,
      seed=None, dt=None,
      w_ir=bp.init.KaimingNormal(scale=1.),
      w_rr=bp.init.KaimingNormal(scale=1.),
      w_ro=bp.init.KaimingNormal(scale=1.)
  ):
    super(RNNNet, self).__init__()

    self.tau = 100
    self.num_input = num_input
    self.num_hidden = num_hidden
    self.num_output = num_output
    if dt is None:
      self.alpha = 1
    else:
      self.alpha = dt / self.tau
    self.rng = bm.random.RandomState(seed=seed)

    # input weight
    self.w_ir = bm.TrainVar(bp.init.parameter(w_ir, (num_input, num_hidden)))

    # recurrent weight
    bound = 1 / num_hidden ** 0.5
    self.w_rr = bm.TrainVar(bp.init.parameter(w_rr, (num_hidden, num_hidden)))
    self.b_rr = bm.TrainVar(self.rng.uniform(-bound, bound, num_hidden))

    # readout weight
    self.w_ro = bm.TrainVar(bp.init.parameter(w_ro, (num_hidden, num_output)))
    self.b_ro = bm.TrainVar(self.rng.uniform(-bound, bound, num_output))

    # variables
    self.h = bm.Variable(bm.zeros((1, num_hidden)), batch_axis=0)

  def reset_state(self, batch_size=None):
    self.h.value = bm.zeros((batch_size, self.num_hidden))

  def cell(self, x, h):
    ins = x @ self.w_ir + h @ self.w_rr + self.b_rr
    state = h * (1 - self.alpha) + ins * self.alpha
    return bm.relu(state)

  def readout(self, h):
    return h @ self.w_ro + self.b_ro

  def update(self, sha, x):
    self.h.value = self.cell(x, self.h.value)
    return self.readout(self.h.value)


# Train the recurrent network on the decision-making task
# ---
# Instantiate the network and print information
with bm.training_environment():
  net = RNNNet(num_input=env.observation_space.shape[0],
               num_hidden=64,
               num_output=env.action_space.n,
               dt=env.dt)


def loss(predictions, targets):
  targets = targets.flatten()
  predictions = predictions.reshape((-1, predictions.shape[-1]))
  total_loss = bp.losses.cross_entropy_loss(predictions, targets)
  # Compute performance
  # Only analyze time points when target is not fixation
  indices = bm.asarray(targets > 0, dtype=bm.float_)
  predictions = predictions.argmax(axis=-1).flatten()
  true_labels = (targets == predictions) * indices
  accuracy = bm.sum(true_labels) / bm.sum(indices)
  return total_loss, {'accuracy': accuracy}


def data_generation():
  for _ in range(100):
    inputs, labels = dataset()
    inputs = bm.asarray(np.moveaxis(inputs, 0, 1))
    labels = bm.asarray(np.moveaxis(labels, 0, 1))
    yield inputs, labels


trainer = bp.train.BPTT(net,
                        loss_fun=loss,
                        loss_has_aux=True,
                        optimizer=bp.optim.Adam(lr=1e-3))
trainer.fit(data_generation, num_epoch=20, num_report=100)

# Visualize neural activity for in sample trials
# ---
# We will run the network for 100 sample trials, then visual the neural activity trajectories in a PCA space.
runner = bp.DSTrainer(net, monitors={'r': net.h}, progress_bar=False)

env.reset(no_step=True)
num_trial = 100
activity_dict = {}
trial_infos = {}
for i in range(num_trial):
  env.new_trial()
  inputs = bm.asarray(env.ob[np.newaxis])
  _ = runner.predict(inputs)
  activity_dict[i] = runner.mon['r'][0]
  trial_infos[i] = env.trial

# Concatenate activity for PCA
activity = np.concatenate(list(activity_dict[i] for i in range(num_trial)), axis=0)
print('Shape of the neural activity: (Time points, Neurons): ', activity.shape)

pca = PCA(n_components=2)
pca.fit(activity)

# Transform individual trials and Visualize in PC space based on ground-truth color. We see that the neural activity is organized by stimulus ground-truth in PC1
plt.rcdefaults()
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 5))
for i in range(num_trial):
  activity_pc = pca.transform(activity_dict[i])
  color = 'red' if trial_infos[i]['ground_truth'] == 0 else 'blue'
  _ = ax1.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)
  if i < 5:
    _ = ax2.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
plt.show()

# Search for approximate fixed points
# ----

net.reset_state(1)  # reset the model first. Analyzer requires batch_size=1
finder = bp.analysis.SlowPointFinder(
  f_cell=net,
  target_vars={'h': net.h},
  args=(bm.asarray([1, 0.5, 0.5]),)
)
fp_candidates = bm.vstack([activity_dict[i] for i in range(num_trial)])
finder.find_fps_with_gd_method(
  candidates={'h': fp_candidates},
  tolerance=1e-5,
  num_batch=200,
  num_opt=int(2e4),
  optimizer=bp.optim.Adam(lr=bp.optim.ExponentialDecay(0.01, 2, 0.9999)),
)
finder.filter_loss(tolerance=1e-5)
finder.keep_unique(tolerance=0.005)

# Visualize the found approximate fixed points.
# ---
# Plot in the same space as activity
plt.figure(figsize=(10, 5))
for i in range(10):
  activity_pc = pca.transform(activity_dict[i])
  trial = trial_infos[i]
  color = 'red' if trial['ground_truth'] == 0 else 'blue'
  plt.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color, alpha=0.1)
# Fixed points are shown in cross
fixedpoints_pc = pca.transform(finder.fixed_points['h'])
plt.plot(fixedpoints_pc[:, 0], fixedpoints_pc[:, 1], 'x', label='fixed points')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.show()

# Computing the Jacobian and Plot distribution of eigenvalues
# ---
finder.compute_jacobians({'h': finder._fixed_points['h'][:20]}, plot=True, num_col=5)
