# %% [markdown]
# # *(Yang, 2020)*: Dynamical system analysis for RNN

# %% [markdown]
# Implementation of the paper:
#
# - Yang G R, Wang X J. Artificial neural networks for neuroscientists: A primer[J]. Neuron, 2020, 107(6): 1048-1070.
#
# The original implementation is based on PyTorch: https://github.com/gyyang/nn-brain/blob/master/RNN%2BDynamicalSystemAnalysis.ipynb

# %%
import brainpy as bp
import brainpy.math as bm
bp.math.set_platform('cpu')

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# %% [markdown]
# In this tutorial, we will use supervised learning to train a recurrent neural network on a simple perceptual decision making task, and analyze the trained network using dynamical system analysis.

# %% [markdown]
# ## Defining a cognitive task

# %%
# We will import the task from the neurogym library.
# Please install neurogym:
# 
# https://github.com/neurogym/neurogym

import neurogym as ngym

# %%
# Environment
task = 'PerceptualDecisionMaking-v0'
kwargs = {'dt': 100}
seq_len = 100

# Make supervised dataset
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                       seq_len=seq_len)

# A sample environment from dataset
env = dataset.env
# Visualize the environment with 2 sample trials
_ = ngym.utils.plot_env(env, num_trials=2, fig_kwargs={'figsize': (8, 6)})

# %%
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
batch_size = dataset.batch_size


# %% [markdown]
# ## Define a vanilla continuous-time recurrent network

# %% [markdown]
# Here we will define a continuous-time neural network but discretize it in time using the Euler method.
# \begin{align}
#     \tau \frac{d\mathbf{r}}{dt} = -\mathbf{r}(t) + f(W_r \mathbf{r}(t) + W_x \mathbf{x}(t) + \mathbf{b}_r).
# \end{align}
#
# This continuous-time system can then be discretized using the Euler method with a time step of $\Delta t$, 
# \begin{align}
#     \mathbf{r}(t+\Delta t) = \mathbf{r}(t) + \Delta \mathbf{r} = \mathbf{r}(t) + \frac{\Delta t}{\tau}[-\mathbf{r}(t) + f(W_r \mathbf{r}(t) + W_x \mathbf{x}(t) + \mathbf{b}_r)].
# \end{align}

# %%
class RNN(bp.dyn.DynamicalSystem):
  def __init__(self, num_input, num_hidden, num_output, num_batch, dt=None, seed=None,
               w_ir=bp.init.KaimingNormal(scale=1.),
               w_rr=bp.init.KaimingNormal(scale=1.),
               w_ro=bp.init.KaimingNormal(scale=1.)):
    super(RNN, self).__init__()

    # parameters
    self.tau = 100
    self.num_batch = num_batch
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
    self.h = bm.Variable(bm.zeros((num_batch, num_hidden)))
    self.o = bm.Variable(bm.zeros((num_batch, num_output)))

  def cell(self, x, h):
    ins = x @ self.w_ir + h @ self.w_rr + self.b_rr
    state = h * (1 - self.alpha) + ins * self.alpha
    return bm.relu(state)

  def readout(self, h):
    return h @ self.w_ro + self.b_ro

  def make_update(self, h: bm.JaxArray, o: bm.JaxArray):
    def f(x):
      h.value = self.cell(x, h.value)
      o.value = self.readout(h.value)

    return f

  def predict(self, xs):
    self.h[:] = 0.
    f = bm.make_loop(self.make_update(self.h, self.o),
                     dyn_vars=self.vars(),
                     out_vars=[self.h, self.o])
    return f(xs)

  def loss(self, xs, ys):
    hs, os = self.predict(xs)
    os = os.reshape((-1, os.shape[-1]))
    loss = bp.losses.cross_entropy_loss(os, ys.flatten())
    return loss, os


# %% [markdown]
# ## Train the recurrent network on the decision-making task

# %%
# Instantiate the network and print information
hidden_size = 64
net = RNN(num_input=input_size,
          num_hidden=hidden_size,
          num_output=output_size,
          num_batch=batch_size,
          dt=env.dt)

# %%
# prediction method
predict = bm.jit(net.predict, dyn_vars=net.vars())

# Adam optimizer
opt = bp.optimizers.Adam(lr=0.001, train_vars=net.train_vars().unique())

# gradient function
grad_f = bm.grad(net.loss,
                 dyn_vars=net.vars(),
                 grad_vars=net.train_vars().unique(),
                 return_value=True,
                 has_aux=True)

# training function
@bm.jit
@bm.function(nodes=(net, opt))
def train(xs, ys):
  grads, (loss, os) = grad_f(xs, ys)
  opt.update(grads)
  return loss, os


# %%
running_acc = 0
running_loss = 0
for i in range(1500):
  inputs, labels_np = dataset()
  inputs = bm.asarray(inputs)
  labels = bm.asarray(labels_np)
  loss, outputs = train(inputs, labels)
  running_loss += loss
  # Compute performance
  output_np = np.argmax(outputs.numpy(), axis=-1).flatten()
  labels_np = labels_np.flatten()
  ind = labels_np > 0  # Only analyze time points when target is not fixation
  running_acc += np.mean(labels_np[ind] == output_np[ind])
  if i % 100 == 99:
    running_loss /= 100
    running_acc /= 100
    print('Step {}, Loss {:0.4f}, Acc {:0.3f}'.format(i + 1, running_loss, running_acc))
    running_loss = 0
    running_acc = 0

# %% [markdown]
# ## Visualize neural activity for in sample trials
#
# We will run the network for 100 sample trials, then visual the neural activity trajectories in a PCA space.

# %%
env.reset(no_step=True)
perf = 0
num_trial = 100
activity_dict = {}
trial_infos = {}
for i in range(num_trial):
    env.new_trial()
    ob, gt = env.ob, env.gt
    inputs = bm.asarray(ob[:, np.newaxis, :])
    rnn_activity, action_pred = predict(inputs)
    rnn_activity = rnn_activity.numpy()[:, 0, :]
    activity_dict[i] = rnn_activity
    trial_infos[i] = env.trial
    
# Concatenate activity for PCA
activity = np.concatenate(list(activity_dict[i] for i in range(num_trial)), axis=0)
print('Shape of the neural activity: (Time points, Neurons): ', activity.shape)

# Print trial informations
for i in range(5):
    print('Trial ', i, trial_infos[i])

# %%
pca = PCA(n_components=2)
pca.fit(activity)

# %% [markdown]
# Transform individual trials and Visualize in PC space based on ground-truth color. We see that the neural activity is organized by stimulus ground-truth in PC1

# %%
plt.rcdefaults()
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 5))
for i in range(num_trial):
    activity_pc = pca.transform(activity_dict[i])
    trial = trial_infos[i]
    color = 'red' if trial['ground_truth'] == 0 else 'blue'
    _ = ax1.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)
    if i < 5:
        _ = ax2.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)

ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
plt.show()

# %% [markdown]
# ## Search for approximate fixed points

# %% [markdown]
# Here we search for approximate fixed points and visualize them in the same PC space. In a generic dynamical system,
# \begin{align}
#     \frac{d\mathbf{x}}{dt} = F(\mathbf{x}),
# \end{align}
# We can search for fixed points by doing the optimization
# \begin{align}
#     \mathrm{argmin}_{\mathbf{x}} |F(\mathbf{x})|^2.
# \end{align}

# %%
f_cell = lambda h: net.cell(bm.asarray([1, 0.5, 0.5]), h)

# %%
fp_candidates = bm.vstack([activity_dict[i] for i in range(num_trial)])
fp_candidates.shape

# %%
finder = bp.analysis.SlowPointFinder(f_cell=f_cell, f_type=bp.analysis.DISCRETE)
finder.find_fps_with_gd_method(
  candidates=fp_candidates,
  tolerance=1e-5, num_batch=200,
  optimizer=bp.optim.Adam(lr=bp.optim.ExponentialDecay(0.01, 1, 0.9999)),
)
finder.filter_loss(tolerance=1e-5)
finder.keep_unique(tolerance=0.03)
finder.exclude_outliers(0.1)
fixed_points = finder.fixed_points

# %% [markdown]
# ## Visualize the found approximate fixed points.
#
# We see that they found an approximate line attrator, corresponding to our PC1, along which evidence is integrated during the stimulus period.

# %%
# Plot in the same space as activity
plt.figure(figsize=(10, 5))
for i in range(10):
    activity_pc = pca.transform(activity_dict[i])
    trial = trial_infos[i]
    color = 'red' if trial['ground_truth'] == 0 else 'blue'
    plt.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color, alpha=0.1)

# Fixed points are shown in cross
fixedpoints_pc = pca.transform(fixed_points)
plt.plot(fixedpoints_pc[:, 0], fixedpoints_pc[:, 1], 'x', label='fixed points')

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.show()

# %% [markdown]
# ## Computing the Jacobian and finding the line attractor

# %%
from jax import jacobian

# %%
dFdh = jacobian(f_cell)(fixed_points[10])

eigval, eigvec = np.linalg.eig(dFdh.numpy())

# %%
# Plot distribution of eigenvalues in a 2-d real-imaginary plot
plt.figure()
plt.scatter(np.real(eigval), np.imag(eigval))
plt.plot([1, 1], [-1, 1], '--')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
