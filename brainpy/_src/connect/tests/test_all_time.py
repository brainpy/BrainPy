import time
from datetime import datetime

import pytest

import brainpy as bp

pytest.skip('skip.', allow_module_level=True)

try:
  import pandas as pd

  df = pd.DataFrame(columns=['connector name', 'superclass', 'connect matrix size',
                             'build function', 'other parameter', 'time(ms)'])
except (ImportError, ModuleNotFoundError):
  print('No pandas installed, skip test.')

# size_same = [100, 500, 2500, 12500, 25000, 37500, 50000]
# size_same = [100, 500, 2500, 12500]
# size_diff = [(10, 100), (100, 1000), (1000, 10000)]

size_same = [100, 500, 2500]
size_diff = [(10, 100), (100, 1000)]


def get_ms(value):
  return round(value * 1000, 4)


def insert_row(connector_name, superclass, connect_matrix_size, build_function, other_parameter, time_used):
  try:
    df.loc[len(df)] = [connector_name, superclass, connect_matrix_size, build_function, other_parameter, time_used]
  except (NameError, UnboundLocalError):
    print('No pandas installed, skip test.')


class OneEndConnector(unittest.TestCase):
  def test_gaussian_prob(self):
    print()
    for size in size_same:
      print('GaussianProb:', size)
      conn = bp.connect.GaussianProb(sigma=1., include_self=False, seed=123)(pre_size=size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('GaussianProb',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'sigma=1/include_self=False',
                 time_used)

      # start = time.time()
      # conn.require(bp.connect.COO)
      # time_used = get_ms(time.time() - start)
      # df.loc[len(df)] = ['GaussianProb',
      #                    'OneEndConnector',
      #                    f'{size}x{size}',
      #                    'build_coo',
      #                    'sigma=1/include_self=False',
      #                    time_used]

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('GaussianProb',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'sigma=1/include_self=False',
                 time_used)

  def test_grid_four(self):
    print()
    for size in size_same:
      print('GridFour:', size)
      conn = bp.connect.GridFour(include_self=False, periodic_boundary=False)(size, size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('GridFour',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'include_self=False/periodic_boundary=False',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('GridFour',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'include_self=False/periodic_boundary=False',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('GridFour',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'include_self=False/periodic_boundary=False',
                 time_used)

  def test_grid_eight(self):
    print()
    for size in size_same:
      print('GridEight:', size)
      conn = bp.connect.GridEight(include_self=False, periodic_boundary=False)(size, size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('GridEight',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'include_self=False/periodic_boundary=False',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('GridEight',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'include_self=False/periodic_boundary=False',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('GridEight',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'include_self=False/periodic_boundary=False',
                 time_used)

  def test_grid_n(self):
    print()
    for size in size_same:
      print('GridN:', size)
      conn = bp.connect.GridN(include_self=False, periodic_boundary=False, N=2)(size, size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('GridN',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'include_self=False/periodic_boundary=False/N=2',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('GridN',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'include_self=False/periodic_boundary=False/N=2',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('GridN',
                 'OneEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'include_self=False/periodic_boundary=False/N=2',
                 time_used)


class TwoEndConnector(unittest.TestCase):
  def test_fixed_prob(self):
    print()
    for size in size_same:
      print('FixedProb:', size)
      conn = bp.connect.FixedProb(prob=0.1, seed=123)
      conn(pre_size=size, post_size=size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('FixedProb',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'prob=0.1',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('FixedProb',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'prob=0.1',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('FixedProb',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'prob=0.1',
                 time_used)

    for size in size_diff:
      print('FixedProb:', size)
      conn = bp.connect.FixedProb(prob=0.1, seed=123)
      conn(pre_size=size[0], post_size=size[1])

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('FixedProb',
                 'TwoEndConnector',
                 f'{size[0]}x{size[1]}',
                 'build_mat',
                 'prob=0.1',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('FixedProb',
                 'TwoEndConnector',
                 f'{size[0]}x{size[1]}',
                 'build_coo',
                 'prob=0.1',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('FixedProb',
                 'TwoEndConnector',
                 f'{size[0]}x{size[1]}',
                 'build_csr',
                 'prob=0.1',
                 time_used)

  def test_fixed_pre_num(self):
    print()
    for size in size_same:
      print('FixedPreNum:', size)
      conn = bp.connect.FixedPreNum(num=0.4, seed=123)
      conn(pre_size=size, post_size=size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'pre_num=10',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'pre_num=10',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'pre_num=10',
                 time_used)

    for size in size_diff:
      print('FixedPreNum:', size)
      conn = bp.connect.FixedPreNum(num=0.4, seed=123)
      conn(pre_size=size[0], post_size=size[1])

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size[0]}x{size[1]}',
                 'build_mat',
                 'pre_num=10',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size[0]}x{size[1]}',
                 'build_coo',
                 'pre_num=10',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size[0]}x{size[1]}',
                 'build_csr',
                 'pre_num=10',
                 time_used)

  def test_fixed_post_num(self):
    print()
    for size in size_same:
      print('FixedPostNum:', size)
      conn = bp.connect.FixedPostNum(num=10, seed=123)
      conn(pre_size=size, post_size=size)

      start = time.time()
      mat = conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'num=10',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'num=10',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'num=10',
                 time_used)

    for size in size_diff:
      print('FixedPostNum:', size)
      conn = bp.connect.FixedPreNum(num=10, seed=123)
      conn(pre_size=size[0], post_size=size[1])

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size[0]}x{size[1]}',
                 'build_mat',
                 'pre_num=10',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size[0]}x{size[1]}',
                 'build_coo',
                 'pre_num=10',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('FixedPreNum',
                 'TwoEndConnector',
                 f'{size[0]}x{size[1]}',
                 'build_csr',
                 'pre_num=10',
                 time_used)

  def test_prob_dist(self):
    print()
    for size in size_same:
      print('ProbDist:', size)
      conn = bp.connect.ProbDist(dist=1, prob=0.5, pre_ratio=0.3, seed=1234, include_self=True)
      conn(pre_size=size, post_size=size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('ProbDist',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'prob=0.5',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('ProbDist',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'dist=1|prob=0.5|pre_ratio=0.3|include_self=True',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('ProbDist',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'dist=1|prob=0.5|pre_ratio=0.3|include_self=True',
                 time_used)

  def test_small_world(self):
    print()
    for size in size_same:
      print('SmallWorld:', size)
      conn = bp.connect.SmallWorld(num_neighbor=2, prob=0.5, include_self=False)
      conn(pre_size=size, post_size=size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('SmallWorld',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'num_neighbor=2/prob=0.5/include_self=False',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('SmallWorld',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'num_neighbor=2/prob=0.5/include_self=False',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('SmallWorld',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'num_neighbor=2/prob=0.5/include_self=False',
                 time_used)

  def test_scale_free_ba(self):
    print()
    for size in size_same:
      print('ScaleFreeBA:', size)
      conn = bp.connect.ScaleFreeBA(m=2)
      conn(pre_size=size, post_size=size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('ScaleFreeBA',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'm=2',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('ScaleFreeBA',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'm=2',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('ScaleFreeBA',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'm=2',
                 time_used)

  def test_scale_free_ba_dual(self):
    print()
    for size in size_same:
      print('ScaleFreeBADual:', size)
      conn = bp.connect.ScaleFreeBADual(m1=2, m2=3, p=0.4)
      conn(pre_size=size, post_size=size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('ScaleFreeBADual',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'm1=2/m2=3/p=0.4',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('ScaleFreeBADual',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'm1=2/m2=3/p=0.4',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('ScaleFreeBADual',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'm1=2/m2=3/p=0.4',
                 time_used)

  def test_power_law(self):
    print()
    for size in size_same:
      print('PowerLaw:', size)
      conn = bp.connect.PowerLaw(m=3, p=0.4)
      conn(pre_size=size, post_size=size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('PowerLaw',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 'm=3/p=0.4',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('PowerLaw',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 'm=3/p=0.4',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('PowerLaw',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 'm=3/p=0.4',
                 time_used)

  def test_one2one(self):
    print()
    for size in size_same:
      print('One2One:', size)
      conn = bp.connect.One2One()
      conn(pre_size=size, post_size=size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('One2One',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 '',
                 time_used)

      start = time.time()
      conn.require(bp.connect.COO)
      time_used = get_ms(time.time() - start)
      insert_row('One2One',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_coo',
                 '',
                 time_used)

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('One2One',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 '',
                 time_used)

  def test_all2all(self):
    print()
    for size in size_same:
      print('All2All:', size)
      conn = bp.connect.All2All()
      conn(pre_size=size, post_size=size)

      start = time.time()
      conn.require(bp.connect.CONN_MAT)
      time_used = get_ms(time.time() - start)
      insert_row('All2All',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_mat',
                 '',
                 time_used)

      # start = time.time()
      # conn.require(bp.connect.COO)
      # time_used = get_ms(time.time() - start)
      # df.loc[len(df)] = ['All2All',
      #                    'TwoEndConnector',
      #                    f'{size}x{size}',
      #                    'build_coo',
      #                    '',
      #                    time_used]

      start = time.time()
      conn.require(bp.connect.CSR)
      time_used = get_ms(time.time() - start)
      insert_row('All2All',
                 'TwoEndConnector',
                 f'{size}x{size}',
                 'build_csr',
                 '',
                 time_used)


class TestSave(unittest.TestCase):
  def test_save(self):
    try:
      df.to_csv('connector_time_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv',
                index=False)
    except (NameError, UnboundLocalError):
      print('No pandas installed, skip test.')
