# -*- coding: utf-8 -*-

import time
import warnings
from functools import partial

from jax import vmap
import jax.numpy
import numpy as np
from jax.scipy.optimize import minimize

import brainpy.math as bm
from brainpy import optimizers as optim
from brainpy.analysis import utils
from brainpy.errors import AnalyzerError

__all__ = [
  'SlowPointFinder',
]


class SlowPointFinder(object):
  """Find fixed/slow points by numerical optimization.

  This class can help you:

  - optimize to find the closest fixed points / slow points
  - exclude any fixed points whose fixed point loss is above threshold
  - exclude any non-unique fixed points according to a tolerance
  - exclude any far-away "outlier" fixed points

  This model implementation is inspired by https://github.com/google-research/computation-thru-dynamics.

  Parameters
  ----------
  f_cell : callable, function
    The function to compute the recurrent units.
  f_type : str
    The system's type: continuous system or discrete system.

    - 'continuous': continuous derivative function, denotes this is a continuous system, or
    - 'discrete': discrete update function, denotes this is a discrete system.
  f_loss_batch : callable, function
    The function to compute the loss.
  verbose : bool
    Whether print the optimization progress.
  """

  def __init__(self, f_cell, f_type='continuous', f_loss_batch=None, verbose=True):
    self.verbose = verbose
    if f_type not in ['discrete', 'continuous']:
      raise AnalyzerError(f'Only support "continuous" (continuous derivative function) or '
                          f'"discrete" (discrete update function), not {f_type}.')

    # functions
    self.f_cell = f_cell
    if f_loss_batch is None:
      if f_type == 'discrete':
        self.f_loss = bm.jit(lambda h: bm.mean((h - f_cell(h)) ** 2))
        self.f_loss_batch = bm.jit(lambda h: bm.mean((h - vmap(f_cell)(h)) ** 2, axis=1))
      if f_type == 'continuous':
        self.f_loss = bm.jit(lambda h: bm.mean(f_cell(h) ** 2))
        self.f_loss_batch = bm.jit(lambda h: bm.mean((vmap(f_cell)(h)) ** 2, axis=1))

    else:
      self.f_loss_batch = f_loss_batch
      self.f_loss = bm.jit(lambda h: bm.mean(f_cell(h) ** 2))
    self.f_jacob_batch = bm.jit(vmap(bm.jacobian(f_cell)))

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

  def find_fps_with_gd_method(self,
                              candidates,
                              tolerance=1e-5,
                              num_batch=100,
                              num_opt=10000,
                              optimizer=None,
                              opt_setting=None):
    """Optimize fixed points with gradient descent methods.

    Parameters
    ----------
    candidates : jax.ndarray, JaxArray
      The array with the shape of (batch size, state dim) of hidden states
      of RNN to start training for fixed points.
    tolerance: float
      The loss threshold during optimization
    num_opt : int
      The maximum number of optimization.
    num_batch : int
      Print training information during optimization every so often.
    opt_setting: optional, dict
      The optimization settings.

      .. deprecated:: 2.1.2
         Use "optimizer" to set optimization method instead.

    optimizer: optim.Optimizer
      The optimizer instance.

      .. versionadded:: 2.1.2
    """

    # optimization settings
    if opt_setting is None:
      if optimizer is None:
        optimizer = optim.Adam(lr=optim.ExponentialDecay(0.2, 1, 0.9999),
                               beta1=0.9, beta2=0.999, eps=1e-8)
      else:
        assert isinstance(optimizer, optim.Optimizer), (f'Must be an instance of '
                                                        f'{optim.Optimizer.__name__}, '
                                                        f'while we got {type(optimizer)}')
    else:
      warnings.warn('Please use "optimizer" to set optimization method. '
                    '"opt_setting" is deprecated since version 2.1.2. ',
                    DeprecationWarning)

      assert isinstance(opt_setting, dict)
      assert 'method' in opt_setting
      assert 'lr' in opt_setting
      opt_method = opt_setting.pop('method')
      if isinstance(opt_method, str):
        assert opt_method in optim.__dict__
        opt_method = getattr(optim, opt_method)
      assert issubclass(opt_method, optim.Optimizer)
      opt_lr = opt_setting.pop('lr')
      assert isinstance(opt_lr, (int, float, optim.Scheduler))
      opt_setting = opt_setting
      optimizer = opt_method(lr=opt_lr, **opt_setting)

    if self.verbose:
      print(f"Optimizing with {optimizer} to find fixed points:")

    # set up optimization
    fixed_points = bm.Variable(bm.asarray(candidates))
    grad_f = bm.grad(lambda: self.f_loss_batch(fixed_points.value).mean(),
                     grad_vars={'a': fixed_points}, return_value=True)
    optimizer.register_vars({'a': fixed_points})
    dyn_vars = optimizer.vars() + {'_a': fixed_points}

    def train(idx):
      gradients, loss = grad_f()
      optimizer.update(gradients)
      return loss

    @partial(bm.jit, dyn_vars=dyn_vars, static_argnames=('start_i', 'num_batch'))
    def batch_train(start_i, num_batch):
      f = bm.make_loop(train, dyn_vars=dyn_vars, has_return=True)
      return f(bm.arange(start_i, start_i + num_batch))

    # Run the optimization
    opt_losses = []
    do_stop = False
    num_opt_loops = int(num_opt / num_batch)
    for oidx in range(num_opt_loops):
      if do_stop: break
      batch_idx_start = oidx * num_batch
      start_time = time.time()
      (_, losses) = batch_train(start_i=batch_idx_start, num_batch=num_batch)
      batch_time = time.time() - start_time
      opt_losses.append(losses)

      if self.verbose:
        print(f"    "
              f"Batches {batch_idx_start + 1}-{batch_idx_start + num_batch} "
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

  def find_fps_with_opt_solver(self, candidates, opt_method=None):
    """Optimize fixed points with nonlinear optimization solvers.

    Parameters
    ----------
    candidates
    opt_method: function, callable
    """

    assert bm.ndim(candidates) == 2 and isinstance(candidates, (bm.JaxArray, jax.numpy.ndarray))
    if opt_method is None:
      opt_method = lambda f, x0: minimize(f, x0, method='BFGS')
    if self.verbose:
      print(f"Optimizing to find fixed points:")
    f_opt = bm.jit(vmap(lambda x0: opt_method(self.f_loss, x0)))
    res = f_opt(bm.as_device_array(candidates))
    valid_ids = jax.numpy.where(res.success)[0]
    self._fixed_points = np.asarray(res.x[valid_ids])
    self._losses = np.asarray(res.fun[valid_ids])
    self._selected_ids = np.asarray(valid_ids)
    if self.verbose:
      print(f'    '
            f'Found {len(valid_ids)} fixed points from {len(candidates)} initial points.')

  def filter_loss(self, tolerance=1e-5):
    """Filter fixed points whose speed larger than a given tolerance.

    Parameters
    ----------
    tolerance: float
      Discard fixed points with squared speed larger than this value.
    """
    if self.verbose:
      print(f"Excluding fixed points with squared speed above "
            f"tolerance {tolerance}:")
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
    num_fps = self.fixed_points.shape[0]
    fps, keep_ids = utils.keep_unique(self.fixed_points, tolerance=tolerance)
    self._fixed_points = fps
    self._losses = self._losses[keep_ids]
    self._selected_ids = self._selected_ids[keep_ids]
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
