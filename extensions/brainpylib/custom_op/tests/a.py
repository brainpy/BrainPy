import brainpy.math as bm
import brainpy as bp
from jax.abstract_arrays import ShapedArray


def try1():
  def abs_eval(events, indices, indptr, *, weight, post_num):
    return ShapedArray((post_num,), bm.float64)

  def con_compute(outs, ins):
    post_val, = outs
    post_val.fill(0)
    events, indices, indptr, weight, _ = ins
    weight = weight[()]
    for i in range(events.size):
      if events[i]:
        for j in range(indptr[i], indptr[i + 1]):
          index = indices[j]
          post_val[index] += weight

  event_sum = bm.XLACustomOp(eval_shape=abs_eval, con_compute=con_compute)

  events = bm.random.rand(10) < 0.2
  indices, indptr = bp.conn.FixedProb(0.1)(10, 20).require('pre2post')
  print(bm.jit(event_sum, static_argnames=('weight', 'post_num'))(events, indices, indptr, weight=1., post_num=20))


def try2():
  def abs_eval(events, indices, indptr, post_val, weight):
    return post_val

  def con_compute(outs, ins):
    post_val, = outs
    events, indices, indptr, _, weight = ins
    weight = weight[()]
    for i in range(events.size):
      if events[i]:
        for j in range(indptr[i], indptr[i + 1]):
          index = indices[j]
          post_val[index] += weight

  event_sum = bm.XLACustomOp(eval_shape=abs_eval, con_compute=con_compute)

  events = bm.random.rand(10) < 0.2
  indices, indptr = bp.conn.FixedProb(0.1)(10, 20).require('pre2post')
  print(bm.jit(event_sum)(events, indices, indptr, bm.zeros(20), 1.))


if __name__ == '__main__':
  try1()
  # try2()
