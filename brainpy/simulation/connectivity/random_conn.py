# -*- coding: utf-8 -*-

import logging

import numpy as np

from brainpy import tools, math, errors
from brainpy.simulation import utils
from brainpy.simulation.connectivity.base import TwoEndConnector

try:
  import numba
except ModuleNotFoundError:
  numba = None


logger = logging.getLogger('brainpy.simulation.connectivity')


__all__ = [
  'FixedPostNum',
  'FixedPreNum',
  'FixedProb',
  'GaussianProb',
  'GaussianWeight',
  'DOG',
  'SmallWorld',
  'ScaleFreeBA',
  'ScaleFreeBADual',
  'PowerLaw',
]


@tools.numba_jit
def _prob_conn(pre_i, num_post, prob, include_self):
  conn_i = []
  conn_j = []
  if include_self:
    for j in range(num_post):
      if np.random.random() < prob:
        conn_i.append(pre_i)
        conn_j.append(j)
  else:
    for j in range(num_post):
      if np.random.random() < prob:
        if pre_i != j:
          conn_i.append(pre_i)
          conn_j.append(j)
  return conn_i, conn_j

@tools.numba_jit
def _prob_conn2(num_need, num_total, i=0, include_self=False):
  prob = np.random.random(num_total)
  if not include_self and i <= num_total:
    prob[i] = 1.
  return np.argsort(prob)[:num_need]



@tools.numba_jit
def _gaussian_weight(pre_i, pre_width, pre_height, num_post, post_width, post_height,
                     w_max, w_min, sigma, normalize, include_self):
  conn_i = []
  conn_j = []
  conn_w = []

  # get normalized coordination
  pre_coords = (pre_i // pre_width, pre_i % pre_width)
  if normalize:
    pre_coords = (pre_coords[0] / (pre_height - 1) if pre_height > 1 else 1.,
                  pre_coords[1] / (pre_width - 1) if pre_width > 1 else 1.)

  for post_i in range(num_post):
    if (pre_i == post_i) and (not include_self):
      continue

    # get normalized coordination
    post_coords = (post_i // post_width, post_i % post_width)
    if normalize:
      post_coords = (post_coords[0] / (post_height - 1) if post_height > 1 else 1.,
                     post_coords[1] / (post_width - 1) if post_width > 1 else 1.)

    # Compute Euclidean distance between two coordinates
    distance = (pre_coords[0] - post_coords[0]) ** 2
    distance += (pre_coords[1] - post_coords[1]) ** 2
    # get weight and conn
    value = w_max * np.exp(-distance / (2.0 * sigma ** 2))
    if value > w_min:
      conn_i.append(pre_i)
      conn_j.append(post_i)
      conn_w.append(value)
  return conn_i, conn_j, conn_w


@tools.numba_jit
def _gaussian_prob(pre_i, pre_width, pre_height, num_post, post_width, post_height,
                   p_min, sigma, normalize, include_self):
  conn_i = []
  conn_j = []
  conn_p = []

  # get normalized coordination
  pre_coords = (pre_i // pre_width, pre_i % pre_width)
  if normalize:
    pre_coords = (pre_coords[0] / (pre_height - 1) if pre_height > 1 else 1.,
                  pre_coords[1] / (pre_width - 1) if pre_width > 1 else 1.)

  for post_i in range(num_post):
    if (pre_i == post_i) and (not include_self):
      continue

    # get normalized coordination
    post_coords = (post_i // post_width, post_i % post_width)
    if normalize:
      post_coords = (post_coords[0] / (post_height - 1) if post_height > 1 else 1.,
                     post_coords[1] / (post_width - 1) if post_width > 1 else 1.)

    # Compute Euclidean distance between two coordinates
    distance = (pre_coords[0] - post_coords[0]) ** 2
    distance += (pre_coords[1] - post_coords[1]) ** 2
    # get weight and conn
    value = np.exp(-distance / (2.0 * sigma ** 2))
    if value > p_min:
      conn_i.append(pre_i)
      conn_j.append(post_i)
      conn_p.append(value)
  return conn_i, conn_j, conn_p


@tools.numba_jit
def _dog(pre_i, pre_width, pre_height, num_post, post_width, post_height,
         w_max_p, w_max_n, w_min, sigma_p, sigma_n, normalize, include_self):
  conn_i = []
  conn_j = []
  conn_w = []

  # get normalized coordination
  pre_coords = (pre_i // pre_width, pre_i % pre_width)
  if normalize:
    pre_coords = (pre_coords[0] / (pre_height - 1) if pre_height > 1 else 1.,
                  pre_coords[1] / (pre_width - 1) if pre_width > 1 else 1.)

  for post_i in range(num_post):
    if (pre_i == post_i) and (not include_self):
      continue

    # get normalized coordination
    post_coords = (post_i // post_width, post_i % post_width)
    if normalize:
      post_coords = (post_coords[0] / (post_height - 1) if post_height > 1 else 1.,
                     post_coords[1] / (post_width - 1) if post_width > 1 else 1.)

    # Compute Euclidean distance between two coordinates
    distance = (pre_coords[0] - post_coords[0]) ** 2
    distance += (pre_coords[1] - post_coords[1]) ** 2
    # get weight and conn
    value = w_max_p * np.exp(-distance / (2.0 * sigma_p ** 2)) - \
            w_max_n * np.exp(-distance / (2.0 * sigma_n ** 2))
    if np.abs(value) > w_min:
      conn_i.append(pre_i)
      conn_j.append(post_i)
      conn_w.append(value)
  return conn_i, conn_j, conn_w


@tools.numba_jit
def _smallworld_rewire(prob, i, all_j, include_self):
  if np.random.random() < prob:
    non_connected = np.where(all_j == False)[0]
    if len(non_connected) <= 1:
      return -1
    # Enforce no self-loops or multiple edges
    w = np.random.choice(non_connected)
    while (not include_self) and w == i:
      non_connected.remove(w)
      w = np.random.choice(non_connected)
    return w


def _random_subset(seq, m, rng):
  """Return m unique elements from seq.

  This differs from random.sample which can return repeated
  elements if seq holds repeated elements.

  Note: rng is a random.Random or numpy.random.RandomState instance.
  """
  targets = set()
  while len(targets) < m:
    x = rng.choice(seq)
    targets.add(x)
  return targets


class FixedProb(TwoEndConnector):
  """Connect the post-synaptic neurons with fixed probability.

  Parameters
  ----------
  prob : float
      The conn probability.
  include_self : bool
      Whether create (i, i) conn?
  seed : optional, int
      Seed the random generator.
  method : str
    The method used to create the connection.

    - ``matrix``: This method will create a big matrix, then, the connectivity is constructed
      from this matrix :math:`(N_{pre}, N_{post})`. In a large network, this method will
      consume huge memories, including a matrix: :math:`(N_{pre}, N_{post})`, two vectors:
      :math:`2 * N_{pre} * N_{post} * prob`.
    - ``iter``: This method will iteratively build the synaptic connections. It has the
      minimum pressure of memory consuming, only :math:`2 * N_{pre} * N_{post} * prob`
      (``i`` and ``j`` vectors).
  """

  def __init__(self, prob, include_self=True, seed=None, method='matrix'):
    super(FixedProb, self).__init__()
    self.prob = prob
    self.include_self = include_self
    self.seed = seed
    self.rng = math.random.RandomState(seed=seed)
    self.method = method
    if method not in ['matrix', 'iter']:
      raise errors.BrainPyError(f'Only support "matrix" and "iter", while we got "{method}"')

  def __call__(self, pre_size, post_size):
    num_pre, num_post = utils.size2len(pre_size), utils.size2len(post_size)
    self.num_pre, self.num_post = num_pre, num_post

    if self.method == 'matrix':
      prob_mat = self.rng.random(size=(num_pre, num_post))
      if not self.include_self:
        prob_mat = math.fill_diagonal(prob_mat, 1.)
      conn_mat = math.array(prob_mat < self.prob, dtype=math.bool_)
      pre_ids, post_ids = math.where(conn_mat)
      self.conn_mat = math.array(conn_mat)

    else:
      if numba is None:
        logging.warning(f'Creating {self.__class__.__name__} by {self.method} method '
                        f'without numba installation is very slow. We recommend you '
                        f'install numba first.')
      else:
        math.random.seed(self.seed)
      pre_ids, post_ids = [], []
      for i in range(num_pre):
        pres, posts = _prob_conn(i, num_post, self.prob, self.include_self)
        pre_ids.extend(pres)
        post_ids.extend(posts)
    self.pre_ids = math.asarray(pre_ids, dtype=math.int_)
    self.post_ids = math.asarray(post_ids, dtype=math.int_)
    return self


class FixedPreNum(TwoEndConnector):
  """Connect the pre-synaptic neurons with fixed number for each post-synaptic neuron.

  Parameters
  ----------
  num : float, int
      The conn probability (if "num" is float) or the fixed number of
      connectivity (if "num" is int).
  include_self : bool
      Whether create (i, i) conn ?
  seed : None, int
      Seed the random generator.
  method : str
    The method used to create the connection.

    - ``matrix``: This method will create a big matrix, then, the connectivity is constructed
      from this matrix :math:`(N_{pre}, N_{post})`. In a large network, this method will
      consume huge memories, including a matrix: :math:`(N_{pre}, N_{post})`, two vectors:
      :math:`2 * N_{need} * N_{post}`.
    - ``iter``: This method will iteratively build the synaptic connections. It has the
      minimum pressure of memory consuming, only :math:`2 * N_{need} * N_{post}`
      (``i`` and ``j`` vectors).
  """

  def __init__(self, num, include_self=True, seed=None, method='matrix'):
    super(FixedPreNum, self).__init__()
    if isinstance(num, int):
      assert num >= 0, '"num" must be bigger than 0.'
    elif isinstance(num, float):
      assert 0. <= num <= 1., '"num" must be in [0., 1.].'
    else:
      raise ValueError(f'Unknown type: {type(num)}')
    self.num = num
    self.seed = seed
    self.include_self = include_self
    self.rng = math.random.RandomState(seed=seed)
    self.method = method
    if method not in ['matrix', 'iter']:
      raise errors.BrainPyError(f'Only support "matrix" and "iter", while we got "{method}"')

  def __call__(self, pre_size, post_size):
    num_pre, num_post = utils.size2len(pre_size), utils.size2len(post_size)
    self.num_pre, self.num_post = num_pre, num_post
    num = self.num if isinstance(self.num, int) else int(self.num * num_pre)
    assert num <= num_pre, f'"num" must be less than "num_pre", but got {num} > {num_pre}'

    if self.method == 'matrix':
      prob_mat = self.rng.random(size=(num_post, num_pre))
      if not self.include_self:
        prob_mat = math.fill_diagonal(prob_mat, 1.)
      arg_sort = math.argsort(prob_mat, axis=1)[:, :num]
      pre_ids = math.asarray(arg_sort.flatten(), dtype=math.int_)
      post_ids = math.asarray(math.repeat(math.arange(num_post), num), dtype=math.int_)

    else:
      if numba is None:
        logging.warning(f'Creating {self.__class__.__name__} by {self.method} method '
                        f'without numba installation is very slow. We recommend you '
                        f'install numba first.')
      else:
        math.random.seed(self.seed)
      pre_ids, post_ids = [], []
      for i in range(self.num_post):
        pres = _prob_conn2(num_need=num, num_total=num_pre, i=i, include_self=self.include_self)
        posts = np.ones_like(pres, dtype=math.int_) * i
        pre_ids.append(pres)
        post_ids.append(posts)
      pre_ids = np.concatenate(pre_ids)
      post_ids = np.concatenate(post_ids)
    self.pre_ids = math.asarray(pre_ids, dtype=math.int_)
    self.post_ids = math.asarray(post_ids, dtype=math.int_)
    return self


class FixedPostNum(TwoEndConnector):
  """Connect the post-synaptic neurons with fixed number for each pre-synaptic neuron.

  Parameters
  ----------
  num : float, int
      The conn probability (if "num" is float) or the fixed number of
      connectivity (if "num" is int).
  include_self : bool
      Whether create (i, i) conn ?
  seed : None, int
      Seed the random generator.
  method : str
    The method used to create the connection.

    - ``matrix``: This method will create a big matrix, then, the connectivity is constructed
      from this matrix :math:`(N_{pre}, N_{post})`. In a large network, this method will
      consume huge memories, including a matrix: :math:`(N_{pre}, N_{post})`, two vectors:
      :math:`2 * N_{need} * N_{pre}`.
    - ``iter``: This method will iteratively build the synaptic connections. It has the
      minimum pressure of memory consuming, only :math:`2 * N_{need} * N_{pre}`
      (``i`` and ``j`` vectors).
  """

  def __init__(self, num, include_self=True, seed=None, method='matrix'):
    super(FixedPostNum, self).__init__()
    if isinstance(num, int):
      assert num >= 0, '"num" must be bigger than 0.'
    elif isinstance(num, float):
      assert 0. <= num <= 1., '"num" must be in [0., 1.].'
    else:
      raise ValueError(f'Unknown type: {type(num)}')
    self.num = num
    self.seed = seed
    self.include_self = include_self
    self.rng = math.random.RandomState(seed=seed)
    self.method = method
    if method not in ['matrix', 'iter']:
      raise errors.BrainPyError(f'Only support "matrix" and "iter", while we got "{method}"')

  def __call__(self, pre_size, post_size):
    num_pre = utils.size2len(pre_size)
    num_post = utils.size2len(post_size)
    self.num_pre = num_pre
    self.num_post = num_post
    num = self.num if isinstance(self.num, int) else int(self.num * num_post)
    assert num <= num_post, f'"num" must be less than "num_post", but got {num} > {num_post}'

    if self.method == 'matrix':
      prob_mat = self.rng.random(size=(num_pre, num_post))
      if not self.include_self:
        prob_mat = math.fill_diagonal(prob_mat, 1.)
      arg_sort = math.argsort(prob_mat, axis=1)[:, :num]
      post_ids = math.asarray(arg_sort.flatten(), dtype=math.int_)
      pre_ids = math.asarray(math.repeat(math.arange(num_pre), num), dtype=math.int_)

    else:
      if numba is None:
        logging.warning(f'Creating {self.__class__.__name__} by {self.method} method '
                        f'without numba installation is very slow. We recommend you '
                        f'install numba first.')
      else:
        math.random.seed(self.seed)
      pre_ids, post_ids = [], []
      for i in range(self.num_pre):
        posts = _prob_conn2(num_need=num, num_total=num_post, i=i, include_self=self.include_self)
        pres = np.ones_like(posts, dtype=math.int_) * i
        pre_ids.append(pres)
        post_ids.append(posts)
      pre_ids = np.concatenate(pre_ids)
      post_ids = np.concatenate(post_ids)

    self.pre_ids = math.asarray(pre_ids, dtype=math.int_)
    self.post_ids = math.asarray(post_ids, dtype=math.int_)
    return self


class GaussianWeight(TwoEndConnector):
  """Builds a Gaussian conn pattern between the two populations, where
  the weights decay with gaussian function.

  Specifically,

  .. backend::

      w(x, y) = w_{max} \\cdot \\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma^2})

  where :math:`(x, y)` is the position of the pre-synaptic neuron (normalized
  to [0,1]) and :math:`(x_c,y_c)` is the position of the post-synaptic neuron
  (normalized to [0,1]), :math:`w_{max}` is the maximum weight. In order to void
  creating useless synapses, :math:`w_{min}` can be set to restrict the creation
  of synapses to the cases where the value of the weight would be superior
  to :math:`w_{min}`. Default is :math:`0.01 w_{max}`.

  Parameters
  ----------
  sigma : float
      Width of the Gaussian function.
  w_max : float
      The weight amplitude of the Gaussian function.
  w_min : float, None
      The minimum weight value below which synapses are not created (default: 0.01 * `w_max`).
  normalize : bool
      Whether normalize the coordination.
  include_self : bool
      Whether create the conn at the same position.
  """

  def __init__(self, sigma, w_max, w_min=None, normalize=True, include_self=True):
    super(GaussianWeight, self).__init__()
    self.sigma = sigma
    self.w_max = w_max
    self.w_min = w_max * 0.01 if w_min is None else w_min
    self.normalize = normalize
    self.include_self = include_self

  def __call__(self, pre_size, post_size):
    num_pre = utils.size2len(pre_size)
    num_post = utils.size2len(post_size)
    self.num_pre = num_pre
    self.num_post = num_post
    if isinstance(pre_size, int):
      pre_size = (pre_size, )
    if isinstance(post_size, int):
      post_size = (post_size, )
    if len(pre_size) == 1:
      pre_size = (1, pre_size)
    if len(post_size) == 1:
      post_size = (1, post_size)
    if not (len(pre_size) == len(post_size) == 2):
      raise errors.BrainPyError(f'{GaussianWeight.__name__} only supports <=2D sizes, '
                                f'while we got pre_size={pre_size}, post_size={post_size}')
    pre_height, pre_width = pre_size
    post_height, post_width = post_size

    if numba is None:
      logging.warning(f'Creating {self.__class__.__name__} without numba installation '
                      f'is very slow. We recommend you install numba first.')
    
    # get the connections and weights
    i, j, w = [], [], []
    for pre_i in range(num_pre):
      a = _gaussian_weight(pre_i=pre_i,
                           pre_width=pre_width,
                           pre_height=pre_height,
                           num_post=num_post,
                           post_width=post_width,
                           post_height=post_height,
                           w_max=self.w_max,
                           w_min=self.w_min,
                           sigma=self.sigma,
                           normalize=self.normalize,
                           include_self=self.include_self)
      i.extend(a[0])
      j.extend(a[1])
      w.extend(a[2])

    self.pre_ids = math.asarray(i, dtype=math.int_)
    self.post_ids = math.asarray(j, dtype=math.int_)
    self.weights = math.asarray(w, dtype=math.float_)
    return self


class GaussianProb(TwoEndConnector):
  """Builds a Gaussian conn pattern between the two populations, where
  the conn probability decay according to the gaussian function.

  Specifically,

  .. backend::

      p=\\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma^2})

  where :math:`(x, y)` is the position of the pre-synaptic neuron
  and :math:`(x_c,y_c)` is the position of the post-synaptic neuron.

  Parameters
  ----------
  sigma : float
      Width of the Gaussian function.
  normalize : bool
      Whether normalize the coordination.
  include_self : bool
      Whether create the conn at the same position.
  seed : bool
      The random seed.
  """

  def __init__(self, sigma, p_min=0., normalize=True, include_self=True, seed=None):
    super(GaussianProb, self).__init__()
    self.sigma = sigma
    self.p_min = p_min
    self.normalize = normalize
    self.include_self = include_self
    self.seed = seed

  def __call__(self, pre_size, post_size):
    if self.seed:
      math.random.seed(self.seed)
    self.num_pre = num_pre = utils.size2len(pre_size)
    self.num_post = num_post = utils.size2len(post_size)
    if isinstance(pre_size, int):
      pre_size = (pre_size,)
    if isinstance(post_size, int):
      post_size = (post_size,)
    if len(pre_size) == 1:
      pre_size = (1, pre_size)
    if len(post_size) == 1:
      post_size = (1, post_size)
    if not (len(pre_size) == len(post_size) == 2):
      raise errors.BrainPyError(f'{GaussianWeight.__name__} only supports <=2D sizes, '
                                f'while we got pre_size={pre_size}, post_size={post_size}')
    pre_height, pre_width = pre_size
    post_height, post_width = post_size
    if numba is None:
      logging.warning(f'Creating {self.__class__.__name__} without numba installation '
                      f'is very slow. We recommend you install numba first.')

    # get the connections
    i, j, p = [], [], []  # conn_i, conn_j, probabilities
    for pre_i in range(num_pre):
      a = _gaussian_prob(pre_i=pre_i,
                         pre_width=pre_width,
                         pre_height=pre_height,
                         num_post=num_post,
                         post_width=post_width,
                         post_height=post_height,
                         p_min=self.p_min,
                         sigma=self.sigma,
                         normalize=self.normalize,
                         include_self=self.include_self)
      i.extend(a[0])
      j.extend(a[1])
      p.extend(a[2])
    p = math.asarray(p, dtype=math.float_)
    selected_idxs = math.where(math.random.random(len(p)) < p)[0]
    self.pre_ids = math.asarray(i, dtype=math.int_)[selected_idxs]
    self.post_ids = math.asarray(j, dtype=math.int_)[selected_idxs]

    # TODO: construct connections by matrix
    return self


class DOG(TwoEndConnector):
  """Builds a Difference-Of-Gaussian (dog) conn pattern between the two populations.

  Mathematically,

  .. backend::

      w(x, y) = w_{max}^+ \\cdot \\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma_+^2})
      -  w_{max}^- \\cdot \\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma_-^2})

  where weights smaller than :math:`0.01 * abs(w_{max} - w_{min})` are not created and
  self-connections are avoided by default (parameter allow_self_connections).

  Parameters
  ----------
  sigmas : tuple
      Widths of the positive and negative Gaussian functions.
  ws_max : tuple
      The weight amplitudes of the positive and negative Gaussian functions.
  w_min : float, None
      The minimum weight value below which synapses are not created
      (default: :math:`0.01 * w_{max}^+ - w_{min}^-`).
  normalize : bool
      Whether normalize the coordination.
  include_self : bool
      Whether create the conn at the same position.
  """

  def __init__(self, sigmas, ws_max, w_min=None, normalize=True, include_self=True):
    super(DOG, self).__init__()
    self.sigma_p, self.sigma_n = sigmas
    self.w_max_p, self.w_max_n = ws_max
    self.w_min = math.abs(ws_max[0] - ws_max[1]) * 0.01 if w_min is None else w_min
    self.normalize = normalize
    self.include_self = include_self

  def __call__(self, pre_size, post_size):
    self.num_pre = num_pre = utils.size2len(pre_size)
    self.num_post = num_post = utils.size2len(post_size)
    if isinstance(pre_size, int):
      pre_size = (pre_size,)
    if isinstance(post_size, int):
      post_size = (post_size,)
    if len(pre_size) == 1:
      pre_size = (1, pre_size)
    if len(post_size) == 1:
      post_size = (1, post_size)
    if not (len(pre_size) == len(post_size) == 2):
      raise errors.BrainPyError(f'{GaussianWeight.__name__} only supports <=2D sizes, while '
                                f'we got pre_size={pre_size}, post_size={post_size}')
    pre_height, pre_width = pre_size
    post_height, post_width = post_size

    # get the connections and weights
    i, j, w = [], [], []  # conn_i, conn_j, weights
    for pre_i in range(num_pre):
      a = _dog(pre_i=pre_i,
               pre_width=pre_width,
               pre_height=pre_height,
               num_post=num_post,
               post_width=post_width,
               post_height=post_height,
               w_max_p=self.w_max_p,
               w_max_n=self.w_max_n,
               w_min=self.w_min,
               sigma_p=self.sigma_p,
               sigma_n=self.sigma_n,
               normalize=self.normalize,
               include_self=self.include_self)
      i.extend(a[0])
      j.extend(a[1])
      w.extend(a[2])

    # format connections and weights
    self.pre_ids = math.asarray(i, dtype=math.int_)
    self.post_ids = math.asarray(j, dtype=math.int_)
    self.weights = math.asarray(w, dtype=math.float_)
    return self


class SmallWorld(TwoEndConnector):
  """Build a Watts–Strogatz small-world graph.

  Parameters
  ----------
  num_neighbor : int
      Each node is joined with its `k` nearest neighbors in a ring
      topology.
  prob : float
      The probability of rewiring each edge
  directed : bool
      Whether the graph is a directed graph.
  include_self : bool
      Whether include the node self.

  Notes
  -----
  First create a ring over $num\\_node$ nodes [1]_.  Then each node in the ring is
  joined to its $num\\_neighbor$ nearest neighbors (or $num\\_neighbor - 1$ neighbors
  if $num\\_neighbor$ is odd). Then shortcuts are created by replacing some edges as
  follows: for each edge $(u, v)$ in the underlying "$num\\_node$-ring with
  $num\\_neighbor$ nearest neighbors" with probability $prob$ replace it with a new
  edge $(u, w)$ with uniformly random choice of existing node $w$.

  References
  ----------
  .. [1] Duncan J. Watts and Steven H. Strogatz,
         Collective dynamics of small-world networks,
         Nature, 393, pp. 440--442, 1998.
  """

  def __init__(self, num_neighbor, prob, directed=False, include_self=False):
    super(SmallWorld, self).__init__()
    self.prob = prob
    self.directed = directed
    self.num_neighbor = num_neighbor
    self.include_self = include_self

  def __call__(self, pre_size, post_size):
    assert pre_size == post_size
    if isinstance(pre_size, int) or (isinstance(pre_size, (tuple, list)) and len(pre_size) == 1):
      num_node = pre_size[0]
      self.num_pre = self.num_post = num_node

      if self.num_neighbor > num_node:
        raise ValueError("num_neighbor > num_node, choose smaller num_neighbor or larger num_node")
      # If k == n, the graph is complete not Watts-Strogatz
      if self.num_neighbor == num_node:
        conn = math.ones((num_node, num_node), dtype=bool)
      else:
        conn = math.zeros((num_node, num_node), dtype=bool)
        nodes = math.array(list(range(num_node)))  # nodes are labeled 0 to n-1
        # connect each node to k/2 neighbors
        for j in range(1, self.num_neighbor // 2 + 1):
          targets = math.concatenate([nodes[j:], nodes[0:j]])  # first j nodes are now last in list
          conn[nodes, targets] = True
          conn[targets, nodes] = True

        # rewire edges from each node
        # loop over all nodes in order (label) and neighbors in order (distance)
        # no self loops or multiple edges allowed
        for j in range(1, self.num_neighbor // 2 + 1):  # outer loop is neighbors
          targets = math.concatenate([nodes[j:], nodes[0:j]])  # first j nodes are now last in list
          if self.directed:
            # inner loop in node order
            for u, v in zip(nodes, targets):
              w = _smallworld_rewire(prob=self.prob, i=u, all_j=conn[u],
                                     include_self=self.include_self)
              if w != -1:
                conn[u, v] = False
                conn[u, w] = True
              w = _smallworld_rewire(prob=self.prob, i=u, all_j=conn[:, u],
                                     include_self=self.include_self)
              if w != -1:
                conn[v, u] = False
                conn[w, u] = True
          else:
            # inner loop in node order
            for u, v in zip(nodes, targets):
              w = _smallworld_rewire(prob=self.prob, i=u, all_j=conn[u],
                                     include_self=self.include_self)
              if w != -1:
                conn[u, v] = False
                conn[v, u] = False
                conn[u, w] = True
                conn[w, u] = True
    else:
      raise NotImplementedError('Currently only support 1D ring connection.')

    self.conn_mat = math.array(conn, dtype=math.bool_)
    pre_ids, post_ids = math.where(conn)
    self.pre_ids = math.asarray(pre_ids, dtype=math.int_)
    self.post_ids = math.asarray(post_ids, dtype=math.int_)
    return self


class ScaleFreeBA(TwoEndConnector):
  """Build a random graph according to the Barabási–Albert preferential
  attachment model.

  A graph of $num\\_node$ nodes is grown by attaching new nodes each with
  $m$ edges that are preferentially attached to existing nodes
  with high degree.

  Parameters
  ----------
  m : int
      Number of edges to attach from a new node to existing nodes
  seed : integer, random_state, or None (default)
      Indicator of random number generation state.

  Raises
  ------
  ValueError
      If `m` does not satisfy ``1 <= m < n``.

  References
  ----------
  .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
         random networks", Science 286, pp 509-512, 1999.
  """

  def __init__(self, m, directed=False, seed=None):
    super(ScaleFreeBA, self).__init__()
    self.m = m
    self.directed = directed
    self.seed = seed
    self.rng = math.random.RandomState(seed)

  def __call__(self, pre_size, post_size=None):
    num_node = utils.size2len(pre_size)
    assert num_node == utils.size2len(post_size)
    self.num_pre = self.num_post = num_node

    if self.m < 1 or self.m >= num_node:
      raise ValueError(f"Barabási–Albert network must have m >= 1 and "
                       f"m < n, while m = {self.m} and n = {num_node}")

    # Add m initial nodes (m0 in barabasi-speak)
    conn = math.zeros((num_node, num_node), dtype=bool)
    # Target nodes for new edges
    targets = list(range(self.m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = self.m
    while source < num_node:
      # Add edges to m nodes from the source.
      origins = [source] * self.m
      conn[origins, targets] = True
      if not self.directed:
        conn[targets, origins] = True
      # Add one node to the list for each new edge just created.
      repeated_nodes.extend(targets)
      # And the new node "source" has m edges to add to the list.
      repeated_nodes.extend([source] * self.m)
      # Now choose m unique nodes from the existing nodes
      # Pick uniformly from repeated_nodes (preferential attachment)
      targets = _random_subset(repeated_nodes, self.m, self.rng)
      source += 1

    self.conn_mat = math.array(conn, dtype=math.bool_)
    pre_ids, post_ids = math.where(conn)
    self.pre_ids = math.asarray(pre_ids, dtype=math.int_)
    self.post_ids = math.asarray(post_ids, dtype=math.int_)
    return self


class ScaleFreeBADual(TwoEndConnector):
  """Build a random graph according to the dual Barabási–Albert preferential
  attachment model.

  A graph of $num\\_node$ nodes is grown by attaching new nodes each with either $m_1$
  edges (with probability $p$) or $m_2$ edges (with probability $1-p$) that
  are preferentially attached to existing nodes with high degree.

  Parameters
  ----------
  m1 : int
      Number of edges to attach from a new node to existing nodes with probability $p$
  m2 : int
      Number of edges to attach from a new node to existing nodes with probability $1-p$
  p : float
      The probability of attaching $m_1$ edges (as opposed to $m_2$ edges)
  seed : integer, random_state, or None (default)
      Indicator of random number generation state.

  Raises
  ------
  ValueError
      If `m1` and `m2` do not satisfy ``1 <= m1,m2 < n`` or `p` does not satisfy ``0 <= p <= 1``.

  References
  ----------
  .. [1] N. Moshiri "The dual-Barabasi-Albert model", arXiv:1810.10538.
  """

  def __init__(self, m1, m2, p, directed=False, seed=None):
    self.m1 = m1
    self.m2 = m2
    self.p = p
    self.directed = directed
    self.seed = seed
    self.rng = math.random.RandomState(seed=seed)
    super(ScaleFreeBADual, self).__init__()

  def __call__(self, pre_size, post_size=None):
    num_node = utils.size2len(pre_size)
    assert num_node == utils.size2len(post_size)
    self.num_pre = self.num_post = num_node

    if self.m1 < 1 or self.m1 >= num_node:
      raise ValueError(f"Dual Barabási–Albert network must have m1 >= 1 and m1 < num_node, "
                       f"while m1 = {self.m1} and num_node = {num_node}.")
    if self.m2 < 1 or self.m2 >= num_node:
      raise ValueError(f"Dual Barabási–Albert network must have m2 >= 1 and m2 < num_node, "
                       f"while m2 = {self.m2} and num_node = {num_node}.")
    if self.p < 0 or self.p > 1:
      raise ValueError(f"Dual Barabási–Albert network must have 0 <= p <= 1, while p = {self.p}")

    # Add max(m1,m2) initial nodes (m0 in barabasi-speak)
    conn = math.zeros((num_node, num_node), dtype=bool)
    # Target nodes for new edges
    targets = list(range(max(self.m1, self.m2)))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the remaining nodes.
    source = max(self.m1, self.m2)
    # Pick which m to use first time (m1 or m2)
    if self.rng.random() < self.p:
      m = self.m1
    else:
      m = self.m2
    while source < num_node:
      # Add edges to m nodes from the source.
      origins = [source] * m
      conn[origins, targets] = True
      if not self.directed:
        conn[targets, origins] = True
      # Add one node to the list for each new edge just created.
      repeated_nodes.extend(targets)
      # And the new node "source" has m edges to add to the list.
      repeated_nodes.extend([source] * m)
      # Pick which m to use next time (m1 or m2)
      if self.rng.random() < self.p:
        m = self.m1
      else:
        m = self.m2
      # Now choose m unique nodes from the existing nodes
      # Pick uniformly from repeated_nodes (preferential attachment)
      targets = _random_subset(repeated_nodes, m, self.rng)
      source += 1

    self.conn_mat = math.array(conn, dtype=math.bool_)
    pre_ids, post_ids = math.where(conn)
    self.pre_ids = math.asarray(pre_ids, dtype=math.int_)
    self.post_ids = math.asarray(post_ids, dtype=math.int_)
    return self


class ScaleFreeBAExtended(TwoEndConnector):
  def __init__(self, ):
    raise NotImplementedError


class PowerLaw(TwoEndConnector):
  """Holme and Kim algorithm for growing graphs with powerlaw
  degree distribution and approximate average clustering.

  Parameters
  ----------
  m : int
      the number of random edges to add for each new node
  p : float,
      Probability of adding a triangle after adding a random edge
  seed : integer, random_state, or None (default)
      Indicator of random number generation state.

  Notes
  -----
  The average clustering has a hard time getting above a certain
  cutoff that depends on `m`.  This cutoff is often quite low.  The
  transitivity (fraction of triangles to possible triangles) seems to
  decrease with network size.

  It is essentially the Barabási–Albert (BA) growth model with an
  extra step that each random edge is followed by a chance of
  making an edge to one of its neighbors too (and thus a triangle).

  This algorithm improves on BA in the sense that it enables a
  higher average clustering to be attained if desired.

  It seems possible to have a disconnected graph with this algorithm
  since the initial `m` nodes may not be all linked to a new node
  on the first iteration like the BA model.

  Raises
  ------
  ValueError
      If `m` does not satisfy ``1 <= m <= n`` or `p` does not
      satisfy ``0 <= p <= 1``.

  References
  ----------
  .. [1] P. Holme and B. J. Kim,
         "Growing scale-free networks with tunable clustering",
         Phys. Rev. E, 65, 026107, 2002.
  """

  def __init__(self, m, p, directed=False, seed=None):
    super(PowerLaw, self).__init__()
    self.m = m
    self.p = p
    if self.p > 1 or self.p < 0:
      raise ValueError(f"p must be in [0,1], while p={self.p}")
    self.directed = directed
    self.seed = seed
    self.rng = math.random.RandomState(seed)

  def __call__(self, pre_size, post_size=None):
    num_node = utils.size2len(pre_size)
    assert num_node == utils.size2len(post_size)
    self.num_pre = self.num_post = num_node

    if self.m < 1 or num_node < self.m:
      raise ValueError(f"Must have m>1 and m<n, while m={self.m} and n={num_node}")
    # add m initial nodes (m0 in barabasi-speak)
    conn = math.zeros((num_node, num_node), dtype=bool)
    repeated_nodes = list(range(self.m))  # list of existing nodes to sample from
    # with nodes repeated once for each adjacent edge
    source = self.m  # next node is m
    while source < num_node:  # Now add the other n-1 nodes
      possible_targets = _random_subset(repeated_nodes, self.m, self.rng)
      # do one preferential attachment for new node
      target = possible_targets.pop()
      conn[source, target] = True
      if not self.directed:
        conn[target, source] = True
      repeated_nodes.append(target)  # add one node to list for each new link
      count = 1
      while count < self.m:  # add m-1 more new links
        if self.rng.random() < self.p:  # clustering step: add triangle
          neighbors = math.where(conn[target])[0]
          neighborhood = [
            nbr for nbr in neighbors
            if not conn[source, nbr] and not nbr == source
          ]
          if neighborhood:  # if there is a neighbor without a link
            nbr = self.rng.choice(neighborhood)
            conn[source, nbr] = True  # add triangle
            if not self.directed:
              conn[nbr, source] = True
            repeated_nodes.append(nbr)
            count = count + 1
            continue  # go to top of while loop
        # else do preferential attachment step if above fails
        target = possible_targets.pop()
        conn[source, target] = True
        if not self.directed:
          conn[target, source] = True
        repeated_nodes.append(target)
        count = count + 1

      repeated_nodes.extend([source] * self.m)  # add source node to list m times
      source += 1

    self.conn_mat = math.array(conn, dtype=math.bool_)
    pre_ids, post_ids = math.where(conn)
    self.pre_ids = math.asarray(pre_ids, dtype=math.int_)
    self.post_ids = math.asarray(post_ids, dtype=math.int_)
    return self
