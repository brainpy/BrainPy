# -*- coding: utf-8 -*-

import inspect
import time
from functools import partial

import numpy as np

import brainpy.math as bm
from brainpy.analysis import utils
from brainpy.errors import AnalyzerError

__all__ = [
  'FixedPointFinder',
]


class FixedPointFinder(object):
  """Find fixed points by numerical optimization.

  This class can help you:

  1. add noise to the fixed point candidates
  2. optimize to find the closest fixed points / slow points
  3. exclude any fixed points whose fixed point loss is above threshold
  4. exclude any non-unique fixed points according to a tolerance
  5. exclude any far-away "outlier" fixed points

  Parameters
  ----------
  candidates : jax.ndarray, JaxArray
      The array with the shape of (batch size, state dim) of hidden states
      of RNN to start training for fixed points.
  f_cell : callable, function
    The function to compute the recurrent units.
  f_type : str
    The system's type: continuous system or discrete system.

    - 'df': continuous derivative function, denotes this is a continuous system, or
    - 'F': discrete update function, denotes this is a discrete system.
  f_loss_batch : callable, function
    The function to compute the loss.
  verbose : bool
    Whether print the optimization progress.
  noise : float
    Gaussian noise added to fixed point candidates before optimization.
  """

  def __init__(self, f_cell, candidates, f_type='df', f_loss_batch=None,
               verbose=True, noise=0.):
    self.verbose = verbose

    if f_type not in ['df', 'F']:
      raise AnalyzerError(f'Only support "df" (continuous derivative function) or '
                          f'"F" (discrete update function), not {f_type}.')

    # functions
    self.f_cell = f_cell
    if f_loss_batch is None:
      if f_type == 'F':
        self.f_loss = bm.jit(lambda h: bm.mean((h - bm.vmap(f_cell, auto_infer=False)(h)) ** 2))
        self.f_loss_batch = bm.jit(lambda h: bm.mean((h - bm.vmap(f_cell, auto_infer=False)(h)) ** 2, axis=1))
      if f_type == 'df':
        self.f_loss = bm.jit(lambda h: bm.mean((bm.vmap(f_cell, auto_infer=False)(h)) ** 2))
        self.f_loss_batch = bm.jit(lambda h: bm.mean((bm.vmap(f_cell, auto_infer=False)(h)) ** 2, axis=1))
    else:
      self.f_loss_batch = f_loss_batch
      self.f_loss = bm.jit(lambda h: bm.mean(self.f_loss_batch(h)))
    self.f_jacob_batch = bm.jit(bm.vmap(bm.jacobian(f_cell)))

    # candidates
    self.noise = noise
    self.candidates = candidates
    if self.verbose and self.noise > 0.0:
      print("Adding noise to fixed point candidates.")
      self.candidates += bm.random.randn(*candidates.shape) * np.sqrt(self.noise)

    # essential variables
    self._losses = None
    self._fixed_points = None
    self._selected_ids = None
    self.opt_losses = None

  @property
  def fixed_points(self):
    """The final fixed points found."""
    return self._fixed_points

  @property
  def losses(self):
    """Losses of fixed points."""
    return self._losses

  @property
  def selected_ids(self):
    """The selected ids of candidate points."""
    return self._selected_ids

  def optimize_fixed_points(self, tolerance=1e-5, num_opt_batch=100,
                            num_opt_max=10000, opt_setting=None, ):
    """Optimize fixed points.

    Parameters
    ----------
    tolerance: float
      The loss threshold during optimization
    num_opt_max : int
      The maximum number of optimization.
    num_opt_batch : int
      Print training information during optimization every so often.
    opt_setting: optional, dict
      The optimization settings.
    """
    if self.verbose:
      print("Optimizing to find fixed points:")

    # optimization settings
    if opt_setting is None:
      opt_method = bm.optimizers.Adam
      opt_lr = bm.optimizers.ExponentialDecay(0.2, 1, 0.9999)
      opt_setting = {'beta1': 0.9,
                     'beta2': 0.999,
                     'eps': 1e-8,
                     'name': None}
    else:
      assert isinstance(opt_setting, dict)
      assert 'method' in opt_setting
      assert 'lr' in opt_setting
      opt_method = opt_setting.pop('method')
      if isinstance(opt_method, str):
        assert opt_method in bm.optimizers.__all__
        opt_method = getattr(bm.optimizers, opt_method)
      assert isinstance(opt_method, type)
      if bm.optimizers.Optimizer not in inspect.getmro(opt_method):
        raise ValueError
      opt_lr = opt_setting.pop('lr')
      assert isinstance(opt_lr, (int, float, bm.optimizers.Scheduler))
      opt_setting = opt_setting

    # set up optimization
    fixed_points = bm.Variable(bm.asarray(self.candidates))
    grad_f = bm.grad(lambda: self.f_loss(fixed_points.value).mean(),
                     grad_vars={'a': fixed_points},
                     return_value=True)
    opt = opt_method(train_vars={'a': fixed_points}, lr=opt_lr, **opt_setting)
    dyn_vars = opt.vars() + {'_a': fixed_points}

    def train(idx):
      gradients, loss = grad_f()
      opt.update(gradients)
      return loss

    @partial(bm.jit, dyn_vars=dyn_vars, static_argnames=('start_i', 'num_batch'))
    def batch_train(start_i, num_batch):
      f = bm.make_loop(train, dyn_vars=dyn_vars, has_return=True)
      return f(bm.arange(start_i, start_i + num_batch))

    # Run the optimization
    opt_losses = []
    do_stop = False
    num_opt_loops = int(num_opt_max / num_opt_batch)
    for oidx in range(num_opt_loops):
      if do_stop: break
      batch_idx_start = oidx * num_opt_batch
      start_time = time.time()
      (_, losses) = batch_train(start_i=batch_idx_start, num_batch=num_opt_batch)
      batch_time = time.time() - start_time
      opt_losses.append(losses)

      if self.verbose:
        print(f"    "
              f"Batches {batch_idx_start + 1}-{batch_idx_start + num_opt_batch} "
              f"in {batch_time:0.2f} sec, Training loss {losses[-1]:0.10f}")

      if losses[-1] < tolerance:
        do_stop = True
        if self.verbose:
          print(f'    '
                f'Stop optimization as mean training loss {losses[-1]:0.10f} '
                f'is below tolerance {tolerance:0.10f}.')
    self.opt_losses = bm.concatenate(opt_losses)
    self._losses = np.asarray(self.f_loss_batch(fixed_points))
    self._fixed_points = np.asarray(fixed_points)
    self._selected_ids = np.arange(fixed_points.shape[0])

  def filter_loss(self, tolerance=1e-5):
    """Filter fixed points whose speed larger than a given tolerance.

    Parameters
    ----------
    tolerance: float
      Discard fixed points with squared speed larger than this value.
    """
    if self.verbose:
      print(f"Excluding fixed points with squared speed above "
            f"tolerance {tolerance:0.5f}:")
    num_fps = self.fixed_points.shape[0]
    ids = self._losses < tolerance
    keep_ids = bm.where(ids)[0]
    self._fixed_points = self._fixed_points[ids]
    self._losses = self._losses[keep_ids]
    self._selected_ids = self._selected_ids[keep_ids]
    if self.verbose:
      print(f"    "
            f"Kept {self._fixed_points.shape[0]}/{num_fps} "
            f"fixed points with tolerance under {tolerance}.")

  def keep_unique(self, tolerance=2.5e-2):
    """Filter unique fixed points by choosing a representative within tolerance.

    Parameters
    ----------
    tolerance: float
      Tolerance for determination of identical fixed points.
    """
    if self.verbose:
      print("Excluding non-unique fixed points:")
    if tolerance <= 0.0:
      return
    if self._fixed_points.shape[0] <= 1:
      return

    # If point a and point b are within identical_tol of each other, and the
    # a is first in the list, we keep a.
    all_drop_idxs = []
    distances = np.asarray(utils.euclidean_distance(self._fixed_points))
    num_fps = self._fixed_points.shape[0]
    example_idxs = np.arange(num_fps)
    for fidx in range(num_fps - 1):
      distances_f = distances[fidx, fidx + 1:]
      drop_idxs = example_idxs[fidx + 1:][distances_f <= tolerance]
      all_drop_idxs += list(drop_idxs)
    unique_dropidxs = np.unique(all_drop_idxs)
    keep_ids = np.setdiff1d(example_idxs, unique_dropidxs)
    if keep_ids.shape[0] > 0:
      self._fixed_points = self._fixed_points[keep_ids, :]
    else:
      self._fixed_points = np.array([], dtype=self._fixed_points.dtype)
    self._selected_ids = self._selected_ids[keep_ids]
    self._losses = self._losses[keep_ids]

    if self.verbose:
      print(f"    Kept {self._fixed_points.shape[0]}/{num_fps} unique fixed points "
            f"with uniqueness tolerance {tolerance}.")

  def exclude_outliers(self, tolerance=1e0):
    """Exclude points whose closest neighbor is further than threshold.

    Parameters
    ----------
    tolerance: float
      Any point whose closest fixed point is greater than tol is an outlier.
    """
    if self.verbose:
      print("Excluding outliers:")
    if np.isinf(tolerance):
      return
    if self._fixed_points.shape[0] <= 1:
      return

    # Compute pairwise distances between all fixed points.
    distances = utils.euclidean_distance(self._fixed_points)

    # Find second smallest element in each column of the pairwise distance matrix.
    # This corresponds to the closest neighbor for each fixed point.
    closest_neighbor = np.partition(distances, kth=1, axis=0)[1]

    # Return data with outliers removed and indices of kept datapoints.
    keep_ids = np.where(closest_neighbor < tolerance)[0]
    num_fps = self._fixed_points.shape[0]
    self._fixed_points = self._fixed_points[keep_ids]
    self._selected_ids = self._selected_ids[keep_ids]
    self._losses = self._losses[keep_ids]

    if self.verbose:
      print(f"    "
            f"Kept {keep_ids.shape[0]}/{num_fps} fixed points "
            f"with within outlier tolerance {tolerance}.")

  def compute_jacobians(self, points):
    """Compute the jacobian matrices at the points.

    Parameters
    ----------
    points: np.ndarray, bm.JaxArray, jax.ndarray
      The fixed points with the shape of (num_point, num_dim).

    Returns
    -------
    jacobians : bm.JaxArray
      npoints number of jacobians, np array with shape npoints x dim x dim
    """
    # if len(self.fixed_points) == 0: return
    if bm.ndim(points) == 1:
      points = bm.asarray([points, ])
    assert bm.ndim(points) == 2
    return self.f_jacob_batch(bm.asarray(points))

  def decompose_eigenvalues(self, matrices, sort_by='magnitude', do_compute_lefts=True):
    """Compute the eigenvalues of the matrices.

    Parameters
    ----------
    matrices: np.ndarray, bm.JaxArray, jax.ndarray
      A 3D array with the shape of (num_matrices, dim, dim).
    do_compute_lefts: bool
      Compute the left eigenvectors? Requires a pseudo-inverse call.

    Returns
    -------
    decompositions : list
      A list of dictionaries with sorted eigenvalues components:
      (eigenvalues, right eigenvectors, and left eigenvectors).
    """
    if sort_by == 'magnitude':
      sort_fun = np.abs
    elif sort_by == 'real':
      sort_fun = np.real
    else:
      raise ValueError("Not implemented yet.")

    decompositions = []
    for mat in matrices:
      eig_values, eig_vectors = np.linalg.eig(mat)
      indices = np.flipud(np.argsort(sort_fun(eig_values)))
      L = None
      if do_compute_lefts:
        L = np.linalg.pinv(eig_vectors).T  # as columns
        L = L[:, indices]
      decompositions.append({'eig_values': eig_values[indices],
                             'R': eig_vectors[:, indices],
                             'L': L})
    return decompositions
