# -*- coding: utf-8 -*-

import time
from functools import partial

import numpy as np
from brainpy import errors

try:
  from scipy.spatial.distance import pdist, squareform
except (ModuleNotFoundError, ImportError):
  pdist = squareform = None

try:
  import jax
  import brainpy.math.jax as bm
except (ModuleNotFoundError, ImportError):
  bm = jax = None


__all__ = [
  'FixedPointFinder',
]


class FixedPointFinder(object):
  """Find fixed points by numerical optimization.

  Parameters
  ----------
  f_cell : callable, function
    The function to compute the recurrent units.
  f_loss : callable, function
    The function to compute the loss.
  verbose : bool
    Whether print the optimization progress.
  num_opt_max : int
    The maximum number of optimization.
  num_opt_batch : int
    Print training information during optimization every so often.
  noise : float
    Gaussian noise added to fixed point candidates before optimization.
  tol_opt : float
    Stop optimizing when the average value of the batch is below this value.
  tol_speed : float
    Discard fixed points with squared speed larger than this value.
  tol_unique : float
    Tolerance for determination of identical fixed points.
  tol_outlier : float
    Any point whose closest fixed point is greater than tol is an outlier.
  """

  def __init__(self,
               # necessary functions
               f_cell,
               f_loss,

               # training parameters
               verbose=True,
               num_opt_batch=100,
               num_opt_max=10000,
               opt_setting=None,

               # optimization parameters
               noise=0.,  # gaussian noise amplitude
               tol_opt=1e-5,  # loss threshold during optimization
               tol_speed=1e-5,  # speed threshold for the given FP
               tol_unique=2.5e-2,  # distance threshold to determine identical FPs
               tol_outlier=1e0  # distance threshold with closest FP to determine outlier
               ):
    if pdist is None or squareform is None:
      raise errors.PackageMissingError('Package "scipy" must be installed when the users '
                                       'want to utilize the fixed point finder analysis.')
    if jax is None or bm is None:
      raise errors.PackageMissingError('Package "jax" must be installed when the users '
                                       'want to utilize the fixed point finder analysis.')

    # functions
    self.f_cell = f_cell
    self.f_loss = f_loss
    self.f_cell_batch = jax.jit(jax.vmap(f_cell))
    self.f_loss_batch = jax.jit(jax.vmap(f_loss))
    self.f_jacob_batch = jax.jit(jax.vmap(jax.jacrev(f_cell)))

    # optimization parameters
    self.noise = noise
    self.tol_opt = tol_opt
    self.tol_speed = tol_speed
    self.tol_unique = tol_unique
    self.tol_outlier = tol_outlier

    # training parameters
    self.verbose = verbose
    self.num_opt_batch = num_opt_batch
    self.num_opt_max = num_opt_max
    if opt_setting is None:
      self.opt_setting = {'method': bm.optimizers.Adam,
                          'lr': bm.optimizers.exponential_decay(0.2, 1, 0.9999),
                          'beta1': 0.9,
                          'beta2': 0.999,
                          'eps': 1e-8,
                          'name': None}
    else:
      assert isinstance(opt_setting, dict)
      assert 'method' in opt_setting
      if isinstance(opt_setting['method'], str):
        assert opt_setting['method'] in bm.optimizers.__all__
        opt_setting['method'] = getattr(bm.optimizers, opt_setting['method'])
      assert 'lr' in opt_setting
      self.opt_setting = opt_setting

  def find_fixed_points(self, candidates):
    """Top-level routine to find fixed points, keeping only valid fixed points.

    This function will:

    1. Add noise to the fixed point candidates
    2. Optimize to find the closest fixed points / slow points
    3. Exclude any fixed points whose fixed point loss is above threshold ('fp_tol')
    4. Exclude any non-unique fixed points according to a tolerance ('unique_tol')
    5. Exclude any far-away "outlier" fixed points ('outlier_tol')

    Arguments:
      rnn_fun: one-step update function as a function of hidden state
      candidates: ndarray with shape npoints x ndims
      hyper_params: dict of hyper parameters for fp optimization, including
        tolerances related to keeping fixed points

    Returns:
      4-tuple of (kept fixed points sorted with slowest points first,
        fixed point losses, indicies of kept fixed points, details of
        optimization)"""

    npoints, dim = candidates.shape

    if self.verbose and self.noise > 0.0:
      print("Adding noise to fixed point candidates.")
      candidates += bm.random.randn(npoints, dim) * np.sqrt(self.noise)

    # 2. find fixed points
    if self.verbose: print("Optimizing to find fixed points:")
    fixed_points, opt_losses = self._optimize_fixed_points(candidates)

    # 3. exclude fixed points whose loss is above threshold
    if self.verbose and self.tol_speed < np.inf:
      print(f"Excluding fixed points with squared speed above "
            f"tolerance {self.tol_speed:0.5f}:")
    fixed_points, fp_ids = self._filter_fixed_points_with_tolerance(fixed_points)
    if len(fp_ids) == 0:
      return np.zeros([0, dim]), np.zeros([0]), [], opt_losses

    # 4. exclude the repeated fixed points
    if self.verbose and self.tol_unique > 0.0:
      print("Excluding non-unique fixed points:")
    fixed_points, unique_ids = self._keep_unique_fixed_points(fixed_points)
    if len(unique_ids) == 0:
      return np.zeros([0, dim]), np.zeros([0]), [], opt_losses

    # 5. exclude outliers
    if self.verbose and self.tol_outlier < np.inf:
      print("Excluding outliers:")
    fixed_points, outlier_ids = self._exclude_outliers(fixed_points, 'euclidean')
    if len(outlier_ids) == 0:
      return np.zeros([0, dim]), np.zeros([0]), [], opt_losses

    if self.verbose:
      print('Sorting fixed points with slowest speed first.')
    losses = np.array(self.f_loss_batch(fixed_points))  # came back as jax.interpreters.xla.DeviceArray
    sort_ids = np.argsort(losses)
    fixed_points = fixed_points[sort_ids]
    losses = losses[sort_ids]
    keep_ids = fp_ids[unique_ids[outlier_ids[sort_ids]]]
    return fixed_points, losses, keep_ids, opt_losses

  def _optimize_fixed_points(self, candidates):
    """Find fixed points via optimization.

    Parameters
    ----------
    candidates : jax.ndarray, JaxArray
      The array with the shape of (batch size, state dim) of hidden states
      of RNN to start training for fixed points.

    Returns
    -------
    fps_and_losses : tuple
      A tuple of (the fixed points, the optimization losses).
    """
    fixed_points = bm.Variable(bm.asarray(candidates))

    @partial(bm.jit, dyn_vars=fixed_points, static_argnames=('start_i', 'num_batch'))
    def batch_train(start_i, num_batch):
      grad_f = bm.grad(lambda: self.f_loss_batch(fixed_points.value).mean(),
                       grad_vars={'a': fixed_points}, return_value=True)
      opt = self.opt_setting['method'](
        train_vars={'a': fixed_points}, lr=self.opt_setting['lr'],
        **{k: v for k, v in self.opt_setting.items() if k not in ['method', 'lr']})

      def train(idx):
        gradients, loss = grad_f()
        # print(gradients)
        opt.update(gradients)
        return loss

      f = bm.easy_scan(train, dyn_vars=opt.implicit_variables, has_return=True)
      return f(bm.arange(start_i, start_i + num_batch))

    # Run the optimization
    opt_losses = []
    do_stop = False
    num_opt_loops = int(self.num_opt_max / self.num_opt_batch)
    for oidx in range(num_opt_loops):
      if do_stop: break
      batch_idx_start = oidx * self.num_opt_batch
      start_time = time.time()
      (_, losses) = batch_train(start_i=batch_idx_start, num_batch=self.num_opt_batch)
      batch_time = time.time() - start_time
      opt_losses.append(losses)

      if self.verbose:
        print(f"    "
              f"Batches {batch_idx_start + 1}-{batch_idx_start + self.num_opt_batch} "
              f"in {batch_time:0.2f} sec, Training loss {losses[-1]:0.10f}")

      if losses[-1] < self.tol_opt:
        do_stop = True
        if self.verbose:
          print(f'    '
                f'Stop optimization as mean training loss {losses[-1]:0.10f} '
                f'is below tolerance {self.tol_opt:0.10f}.')
    return fixed_points.numpy(), bm.concatenate(opt_losses).numpy()

  def _filter_fixed_points_with_tolerance(self, fixed_points):
    """Filter fixed points whose speed larger than a given tolerance.

    Parameters
    ----------
    fixed_points: np.ndarray
      The ndarray with shape of (num_point, num_dim).

    Returns
    -------
    fps_and_ids : tuple
      A 2-tuple of (kept fixed points, ids of kept fixed points).
    """
    losses = self.f_loss_batch(fixed_points)
    ids = losses < self.tol_speed
    keep_ids = np.where(ids)[0]
    fps_w_tol = fixed_points[ids]
    if self.verbose:
      print(f"    "
            f"Kept {fps_w_tol.shape[0]}/{fixed_points.shape[0]} "
            f"fixed points with tolerance under {self.tol_speed}.")

    return fps_w_tol, keep_ids

  def _keep_unique_fixed_points(self, fixed_points):
    """Filter unique fixed points by choosing a representative within tolerance.

    Parameters
    ----------
    fixed_points: np.ndarray
      The fixed points with the shape of (num_point, num_dim).

    Returns
    -------
    fps_and_ids : tuple
      A 2-tuple of (kept fixed points, ids of kept fixed points).
    """
    keep_ids = np.arange(fixed_points.shape[0])
    if self.tol_unique <= 0.0:
      return fixed_points, keep_ids
    if fixed_points.shape[0] <= 1:
      return fixed_points, keep_ids

    nfps = fixed_points.shape[0]
    all_drop_idxs = []

    # If point a and point b are within identical_tol of each other, and the
    # a is first in the list, we keep a.
    example_idxs = np.arange(nfps)
    distances = squareform(pdist(fixed_points, metric="euclidean"))
    for fidx in range(nfps - 1):
      distances_f = distances[fidx, fidx + 1:]
      drop_idxs = example_idxs[fidx + 1:][distances_f <= self.tol_unique]
      all_drop_idxs += list(drop_idxs)

    unique_dropidxs = np.unique(all_drop_idxs)
    keep_ids = np.setdiff1d(example_idxs, unique_dropidxs)
    if keep_ids.shape[0] > 0:
      unique_fps = fixed_points[keep_ids, :]
    else:
      unique_fps = np.array([], dtype=np.int64)

    if self.verbose:
      print(f"    Kept {unique_fps.shape[0]}/{nfps} unique fixed points "
            f"with uniqueness tolerance {self.tol_unique}.")

    return unique_fps, keep_ids

  def _exclude_outliers(self, fixed_points, metric='euclidean'):
    """Exclude points whose closest neighbor is further than threshold.

    Parameters
    ----------
    fixed_points: np.ndarray
      The fixed points with the shape of (num_point, num_dim).
    metric: str
      The distance metric passed to scipy.spatial.pdist. Defaults to "euclidean"

    Returns
    -------
    fps_and_ids : tuple
      A 2-tuple of (kept fixed points, ids of kept fixed points).
    """


    if np.isinf(self.tol_outlier):
      return fixed_points, np.arange(len(fixed_points))
    if fixed_points.shape[0] <= 1:
      return fixed_points, np.arange(len(fixed_points))

    # Compute pairwise distances between all fixed points.
    distances = squareform(pdist(fixed_points, metric=metric))

    # Find second smallest element in each column of the pairwise distance matrix.
    # This corresponds to the closest neighbor for each fixed point.
    closest_neighbor = np.partition(distances, kth=1, axis=0)[1]

    # Return data with outliers removed and indices of kept datapoints.
    keep_idx = np.where(closest_neighbor < self.tol_outlier)[0]
    data_to_keep = fixed_points[keep_idx]

    if self.verbose:
      print(f"    "
            f"Kept {data_to_keep.shape[0]}/{fixed_points.shape[0]} fixed points "
            f"with within outlier tolerance {self.tol_outlier}.")

    return data_to_keep, keep_idx

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
    if len(points) == 0: return
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
