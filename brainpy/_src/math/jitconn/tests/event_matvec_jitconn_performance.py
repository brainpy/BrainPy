from time import time

from jax import jit

import brainpy as bp
import brainpy.math as bm


def compare_sparse_ops(platform='cpu'):
  """

  GPU
  ---
  shape = (1000, 1000), prob = 0.1, transpose = True
  csr sparse 0.09568500518798828 s
  jit conn   0.12936949729919434 s

  shape = (1000, 1000), prob = 0.1, transpose = False
  csr sparse 0.09957313537597656 s
  jit conn   0.1456453800201416 s

  shape = (1000, 1000), prob = 0.2, transpose = True
  csr sparse 0.1014559268951416 s
  jit conn   0.16193556785583496 s

  shape = (1000, 1000), prob = 0.2, transpose = False
  csr sparse 0.10938715934753418 s
  jit conn   0.14464354515075684 s

  shape = (1000, 1000), prob = 0.4, transpose = True
  csr sparse 0.14374589920043945 s
  jit conn   0.1551048755645752 s

  shape = (1000, 1000), prob = 0.4, transpose = False
  csr sparse 0.14356279373168945 s
  jit conn   0.15198969841003418 s

  shape = (1000, 1000), prob = 0.6, transpose = True
  csr sparse 0.1429135799407959 s
  jit conn   0.15459179878234863 s

  shape = (1000, 1000), prob = 0.6, transpose = False
  csr sparse 0.14870882034301758 s
  jit conn   0.15899157524108887 s

  shape = (1000, 1000), prob = 0.8, transpose = True
  csr sparse 0.1489548683166504 s
  jit conn   0.1636965274810791 s

  shape = (1000, 1000), prob = 0.8, transpose = False
  csr sparse 0.09073925018310547 s
  jit conn   0.17296433448791504 s

  shape = (1000, 10000), prob = 0.1, transpose = True
  csr sparse 0.14572954177856445 s
  jit conn   0.15570378303527832 s

  shape = (1000, 10000), prob = 0.1, transpose = False
  csr sparse 0.14201974868774414 s
  jit conn   0.2694075107574463 s

  shape = (1000, 10000), prob = 0.2, transpose = True
  csr sparse 0.1480388641357422 s
  jit conn   0.14784669876098633 s

  shape = (1000, 10000), prob = 0.2, transpose = False
  csr sparse 0.14451289176940918 s
  jit conn   0.4144716262817383 s

  shape = (1000, 10000), prob = 0.4, transpose = True
  csr sparse 0.14377927780151367 s
  jit conn   0.15256381034851074 s

  shape = (1000, 10000), prob = 0.4, transpose = False
  csr sparse 0.1487278938293457 s
  jit conn   0.41004467010498047 s

  shape = (1000, 10000), prob = 0.6, transpose = True
  csr sparse 0.1689896583557129 s
  jit conn   0.18367314338684082 s

  shape = (1000, 10000), prob = 0.6, transpose = False
  csr sparse 0.15153169631958008 s
  jit conn   0.4159865379333496 s

  shape = (1000, 10000), prob = 0.8, transpose = True
  csr sparse 0.15267014503479004 s
  jit conn   0.16814088821411133 s

  shape = (1000, 10000), prob = 0.8, transpose = False
  csr sparse 0.1320178508758545 s
  jit conn   0.5114090442657471 s

  shape = (10000, 10000), prob = 0.1, transpose = True
  csr sparse 0.15414834022521973 s
  jit conn   0.15847539901733398 s

  shape = (10000, 10000), prob = 0.1, transpose = False
  csr sparse 0.1557462215423584 s
  jit conn   0.18897342681884766 s

  shape = (10000, 10000), prob = 0.2, transpose = True
  csr sparse 0.28719663619995117 s
  jit conn   0.3945181369781494 s

  shape = (10000, 10000), prob = 0.2, transpose = False
  csr sparse 0.29045557975769043 s
  jit conn   0.2662692070007324 s

  shape = (10000, 10000), prob = 0.4, transpose = True
  csr sparse 0.26814866065979004 s
  jit conn   0.41262269020080566 s

  shape = (10000, 10000), prob = 0.4, transpose = False
  csr sparse 0.14010882377624512 s
  jit conn   0.30821704864501953 s

  shape = (10000, 10000), prob = 0.6, transpose = True
  csr sparse 0.34110474586486816 s
  jit conn   0.44765257835388184 s

  shape = (10000, 10000), prob = 0.6, transpose = False
  csr sparse 0.14516901969909668 s
  jit conn   0.42423462867736816 s

  shape = (10000, 10000), prob = 0.8, transpose = True
  csr sparse 0.38806986808776855 s
  jit conn   0.5052323341369629 s

  shape = (10000, 10000), prob = 0.8, transpose = False
  csr sparse 0.13016152381896973 s
  jit conn   0.4791419506072998 s

  shape = (50000, 50000), prob = 0.1, transpose = True
  csr sparse 0.1485145092010498 s
  jit conn   0.6013796329498291 s

  shape = (50000, 50000), prob = 0.1, transpose = False
  csr sparse 0.2520942687988281 s
  jit conn   0.5886740684509277 s

  shape = (50000, 50000), prob = 0.2, transpose = True
  csr sparse 0.41227173805236816 s
  jit conn   1.0801291465759277 s

  shape = (50000, 50000), prob = 0.2, transpose = False
  csr sparse 0.5962152481079102 s
  jit conn   1.1053071022033691 s

  shape = (50000, 50000), prob = 0.4, transpose = True
  Killed
  """

  bm.set_platform(platform)

  weight = 1.
  seed = 1234

  all_shapes = [
    (int(1e3), int(1e3)),
    (int(1e3), int(1e4)),
    (int(1e4), int(1e4)),
    (int(5e4), int(5e4)),
    (int(5e4), int(1e5)),
  ]

  for shape in all_shapes:
    for prob in [0.1, 0.2, 0.4, 0.6, 0.8]:
      indices, indptr = bp.conn.FixedProb(prob, pre=shape[0], post=shape[1]).require('csr')
      indices = bm.as_jax(indices)
      indptr = bm.as_jax(indptr)
      for transpose in [True, False]:
        print(f'shape = {shape}, prob = {prob}, transpose = {transpose}')
        f_sparse = jit(lambda e: bm.event.csrmv(weight, indices, indptr, e,
                                                shape=shape, transpose=transpose))
        f_jitconn = jit(lambda e: bm.jitconn.event_mv_prob_homo(
          e, weight, conn_prob=prob, shape=shape, seed=seed, transpose=transpose))

        rng = bm.random.RandomState()
        events = rng.random(shape[0] if transpose else shape[1]).value < prob
        f_sparse(events).block_until_ready()
        f_jitconn(events).block_until_ready()

        t0 = time()
        for _ in range(100):
          f_sparse(events).block_until_ready()
        print(f'csr sparse {time() - t0} s')

        t0 = time()
        for _ in range(100):
          f_jitconn(events).block_until_ready()
        print(f'jit conn   {time() - t0} s')

        print()
      bm.clear_buffer_memory()


def compare_jitconn_imp(platform='gpu'):
  bm.set_platform(platform)

  weight = 1.
  seed = 1234

  all_shapes = [
    (int(1e3), int(1e3)),
    (int(1e3), int(1e4)),
    (int(1e4), int(1e4)),
    (int(5e4), int(5e4)),
    (int(5e4), int(1e5)),
    (int(5e5), int(1e5)),
    (int(5e5), int(5e5)),
  ]

  for shape in all_shapes:
    for prob in [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]:
      for transpose in [True, False]:
        print(f'shape = {shape}, prob = {prob}, transpose = {transpose}')
        # f1 = jit(lambda e: event_matvec_prob_conn_homo_weight_v1(
        #   e, weight, conn_prob=prob, shape=shape, seed=seed, transpose=transpose))
        f2 = jit(lambda e: bm.jitconn.event_mv_prob_homo(
          e, weight, conn_prob=prob, shape=shape, seed=seed, transpose=transpose))

        rng = bm.random.RandomState()
        events = rng.random(shape[0] if transpose else shape[1]).value < prob
        # f1(events).block_until_ready()
        f2(events).block_until_ready()

        # t0 = time()
        # for _ in range(100):
        #   f1(events).block_until_ready()
        # print(f'event_matvec_v1 {time() - t0} s')

        t0 = time()
        for _ in range(100):
          f2(events).block_until_ready()
        print(f'event_matvec_v2 {time() - t0} s')
        print()
        bm.clear_buffer_memory()


if __name__ == '__main__':
  pass
  # compare_where('cpu')
  # compare_sparse_ops('gpu')
  # compare_jitconn_imp('gpu')
