# -*- coding: utf-8 -*-
import sys
import tempfile
import unittest
from functools import partial

import jax
from jax import vmap

from absl.testing import parameterized
from jax._src import test_util as jtu

import brainpy as bp
import brainpy.math as bm


class TestLoop(parameterized.TestCase):
  def test_make_loop(self):
    def make_node(v1, v2):
      def update(x):
        v1.value = v1 * x
        return (v1 + v2) * x

      return update

    bp.math.random.seed()
    _v1 = bm.Variable(bm.random.normal(size=10))
    _v2 = bm.Variable(bm.random.random(size=10))
    _xs = bm.random.uniform(size=(4, 10))

    scan_f = bm.make_loop(make_node(_v1, _v2),
                          dyn_vars=(_v1, _v2),
                          out_vars=(_v1,),
                          has_return=True)
    outs, returns = scan_f(_xs)
    for out in outs:
      print(out.shape)
    print(outs)
    print(returns.shape)
    print(returns)

    print('-' * 20)
    scan_f = bm.make_loop(make_node(_v1, _v2),
                          dyn_vars=(_v1, _v2),
                          out_vars=_v1,
                          has_return=True)
    outs, returns = scan_f(_xs)
    print(outs.shape)
    print(outs)
    print(returns.shape)
    print(returns)

    print('-' * 20)
    scan_f = bm.make_loop(make_node(_v1, _v2),
                          dyn_vars=(_v1, _v2),
                          has_return=True)
    outs, returns = scan_f(_xs)
    print(outs)
    print(returns.shape)
    print(returns)

  def test_make_loop_jaxarray(self):
    def make_node(v1, v2):
      def update(x):
        v1.value = v1 * x
        return (v1 + v2) * x

      return update

    bp.math.random.seed()
    _v1 = bm.Variable(bm.random.normal(size=10))
    _v2 = bm.as_variable(bm.random.random(size=10))
    _xs = bm.random.uniform(size=(4, 10))

    scan_f = bm.make_loop(make_node(_v1, _v2),
                          dyn_vars=(_v1, _v2),
                          out_vars=(_v1,),
                          has_return=True)
    # with self.assertRaises(bp.errors.MathError):
    outs, returns = scan_f(_xs)

  @parameterized.named_parameters(
    {"testcase_name": "_jit_scan={}_jit_f={}_unroll={}".format(jit_scan, jit_f, unroll),
     "jit_scan": jit_scan,
     "jit_f": jit_f,
     "unroll": unroll}
    for jit_scan in [False, True]
    for jit_f in [False, True]
    for unroll in [1, 2]
  )
  def test_for_loop(self, jit_scan, jit_f, unroll):
    rng = bm.random.RandomState(123)

    c = bm.Variable(rng.randn(4))
    d = rng.randn(2)
    all_a = rng.randn(5, 3)

    def f(a):
      assert a.shape == (3,)
      assert c.shape == (4,)
      b = bm.cos(bm.sum(bm.sin(a)) + bm.sum(bm.cos(c)) + bm.sum(bm.tan(d)))
      c.value = bm.sin(c * b)
      assert b.shape == ()
      return b

    if jit_f:
      f = bm.jit(f)
    scan = partial(bm.for_loop, f, unroll=unroll, )
    if jit_scan:
      scan = bm.jit(scan)
    ans = scan(operands=all_a)
    print(ans)
    print(c)

  def test_for_loop_progress_bar(self):
    xs = bm.arange(100)
    ys = bm.for_loop(lambda a: a, xs, progress_bar=True)
    self.assertTrue(bm.allclose(xs, ys))

  def test_for_loop2(self):
    class MyClass(bp.DynamicalSystem):
      def __init__(self):
        super().__init__()
        self.a = bm.Variable(bm.zeros(1))

      def update(self):
        self.a += 1

    cls = MyClass()
    indices = bm.arange(10)
    bm.for_loop(cls.step_run, indices)
    self.assertTrue(bm.allclose(cls.a, 10.))


class TestScan(unittest.TestCase):
  def test1(self):
    a = bm.Variable(1)

    def f(carray, x):
      carray += x
      a.value += 1.
      return carray, a

    carry, outs = bm.scan(f, bm.zeros(2), bm.arange(10))
    self.assertTrue(bm.allclose(carry, 45.))
    expected = bm.arange(1, 11).astype(outs.dtype)
    expected = bm.expand_dims(expected, axis=-1)
    self.assertTrue(bm.allclose(outs, expected))


class TestCond(unittest.TestCase):
  def test1(self):
    bm.random.seed(1)
    bm.cond(True, lambda: bm.random.random(10), lambda: bm.random.random(10), ())
    bm.cond(False, lambda: bm.random.random(10), lambda: bm.random.random(10), ())


class TestIfElse(unittest.TestCase):
  def test1(self):
    def f(a):
      return bm.ifelse(conditions=[a < 0, a < 2, a < 5, a < 10],
                       branches=[lambda: 1,
                                 lambda: 2,
                                 lambda: 3,
                                 lambda: 4,
                                 lambda: 5])

    self.assertTrue(f(3) == 3)
    self.assertTrue(f(1) == 2)
    self.assertTrue(f(-1) == 1)

  def test2(self):
    def f(a):
      return bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
                       branches=[1, 2, 3, 4, 5])

    self.assertTrue(f(3) == 3)
    self.assertTrue(f(1) == 4)
    self.assertTrue(f(-1) == 5)

  def test_dyn_vars1(self):
    var_a = bm.Variable(bm.zeros(1))

    def f(a):
      def f1():
        var_a.value += 1
        return 1

      return bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
                       branches=[f1,
                                 lambda: 2, lambda: 3,
                                 lambda: 4, lambda: 5],
                       dyn_vars=var_a,
                       show_code=True)

    self.assertTrue(f(11) == 1)
    print(var_a)
    self.assertTrue(bm.all(var_a == 1))
    self.assertTrue(f(1) == 4)
    self.assertTrue(f(-1) == 5)

  def test_vmap(self):
    def f(operands):
      f = lambda a: bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
                              branches=[lambda _: 1,
                                        lambda _: 2,
                                        lambda _: 3,
                                        lambda _: 4,
                                        lambda _: 5, ],
                              operands=a,
                              show_code=True)
      return vmap(f)(operands)

    r = f(bm.random.randint(-20, 20, 200))
    self.assertTrue(r.size == 200)

  def test_vmap2(self):
    def f2():
      f = lambda a: bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
                              branches=[1, 2, 3, 4, lambda _: 5],
                              operands=a,
                              show_code=True)
      return vmap(f)(bm.random.randint(-20, 20, 200))

    self.assertTrue(f2().size == 200)

  def test_grad1(self):
    def F2(x):
      return bm.ifelse(conditions=(x >= 10,),
                       branches=[lambda x: x,
                                 lambda x: x ** 2, ],
                       operands=x)

    self.assertTrue(bm.grad(F2)(9.0) == 18.)
    self.assertTrue(bm.grad(F2)(11.0) == 1.)


  def test_grad2(self):
    def F3(x):
      return bm.ifelse(conditions=(x >= 10, x >= 0),
                       branches=[lambda x: x,
                                 lambda x: x ** 2,
                                 lambda x: x ** 4, ],
                       operands=x)

    self.assertTrue(bm.grad(F3)(9.0) == 18.)
    self.assertTrue(bm.grad(F3)(11.0) == 1.)


class TestWhile(unittest.TestCase):
  def test1(self):
    bm.random.seed()

    a = bm.Variable(bm.zeros(1))
    b = bm.Variable(bm.ones(1))

    def cond(x, y):
      return x < 6.

    def body(x, y):
      a.value += x
      b.value *= y
      return x + b[0], y + 1.

    res = bm.while_loop(body, cond, operands=(1., 1.))
    print()
    print(res)

  def test2(self):
    bm.random.seed()

    a = bm.Variable(bm.zeros(1))
    b = bm.Variable(bm.ones(1))

    def cond(x, y):
      return x < 6.

    def body(x, y):
      a.value += x
      b.value *= y
      return x + b[0], y + 1.

    res = bm.while_loop(body, cond, operands=(1., 1.))
    print()
    print(res)

    with jax.disable_jit():
      a = bm.Variable(bm.zeros(1))
      b = bm.Variable(bm.ones(1))

      res2 = bm.while_loop(body, cond, operands=(1., 1.))
      print(res2)
      self.assertTrue(bm.array_equal(res2[0], res[0]))
      self.assertTrue(bm.array_equal(res2[1], res[1]))

  def test3(self):
    bm.random.seed()

    a = bm.Variable(bm.zeros(1))
    b = bm.Variable(bm.ones(1))

    def cond(x, y):
      return bm.all(a.value < 6.)

    def body(x, y):
      a.value += x
      b.value *= y

    res = bm.while_loop(body, cond, operands=(1., 1.))
    self.assertTrue(bm.allclose(a, 6.))
    self.assertTrue(bm.allclose(b, 1.))
    print()
    print(res)
    print(a)
    print(b)

  def test4(self):
    bm.random.seed()

    a = bm.Variable(bm.zeros(1))
    b = bm.Variable(bm.ones(1))

    def cond(x, y):
      a.value += 1
      return bm.all(a.value < 6.)

    def body(x, y):
      a.value += x
      b.value *= y

    res = bm.while_loop(body, cond, operands=(1., 1.))
    self.assertTrue(bm.allclose(a, 5.))
    self.assertTrue(bm.allclose(b, 1.))
    print(res)
    print(a)
    print(b)
    print()

  def test5(self):
    bm.random.seed()

    a = bm.Variable(bm.zeros(1))
    b = bm.Variable(bm.ones(1))
    c = bm.Variable(bm.ones(1))

    def cond(x, y):
      a.value += 1
      return bm.all(a.value < 6.)

    def body(x, y):
      a.value += x
      b.value *= y
      return x + 1, y + 1

    @bm.jit
    def run(a, b):
      x, y = bm.while_loop(body, cond, operands=(a, b))
      return c + x

    run(0., 1.)

    # self.assertTrue(bm.allclose(a, 5.))
    # self.assertTrue(bm.allclose(b, 1.))
    # print(a)
    # print(b)
    # print()


class TestDebugAndCompile(parameterized.TestCase):
  def test_cond1(self):
    file = tempfile.TemporaryFile('w+')

    def f_true(a):
      print('True function ..', file=file)
      return a

    def f_false(a):
      print('False function ..', file=file)
      return a

    jax.lax.cond(True, f_true, f_false, 1.)

    expected_res = '''
True function ..
False function ..
    '''

    file.seek(0)
    self.assertTrue(file.read().strip() == expected_res.strip())

  def test_cond2(self):
    file = tempfile.TemporaryFile('w+')

    def f1(a):
      print('f1 ...', file=file)
      return a * 0.1

    def f2(a):
      print('f2 ...', file=file)
      return a * 1.

    def f3(a):
      print('f3 ...', file=file)
      return bm.cond(a > 1, f1, f2, a)

    def f4(a):
      print('f4 ...', file=file)
      return a * 10.

    r = bm.cond(True, f3, f4, 2.)
    print(r)

    expected_res = '''
f3 ...
f1 ...
f2 ...
f4 ...
f3 ...
f1 ...
f2 ...
f4 ...
    '''
    file.seek(0)
    # print(file.read().strip())
    self.assertTrue(file.read().strip() == expected_res.strip())

  def test_for_loop(self):
    def f(a):
      print('f ...', file=file)
      return a

    file = tempfile.TemporaryFile('w+')
    bm.for_loop(f, bm.arange(10))
    file.seek(0)
    expect = '''
f ...
f ...
    '''
    self.assertTrue(file.read().strip() == expect.strip())

    file = tempfile.TemporaryFile('w+')
    bm.for_loop(f, bm.arange(10), jit=False)
    file.seek(0)
    expect = '\n'.join(['f ...'] * 10)
    self.assertTrue(file.read().strip() == expect.strip())

  def test_while_loop(self):
    def cond(a):
      print('cond ...', file=file)
      return a < 1

    def body(a):
      print('body ...', file=file)
      return a + 1

    file = tempfile.TemporaryFile('w+')
    bm.while_loop(body, cond, 10)
    file.seek(0)
    expect = '''
cond ...
body ...
cond ...
body ...
    '''
    out1 = file.read().strip()
    print(out1)
    self.assertTrue(out1 == expect.strip())

    file = tempfile.TemporaryFile('w+')
    jax.lax.while_loop(cond, body, 10)
    file.seek(0)
    out2 = file.read().strip()
    expect = '''
cond ...
body ...
    '''
    self.assertTrue(expect.strip() == out2)

    file = tempfile.TemporaryFile('w+')
    with jax.disable_jit():
      jax.lax.while_loop(cond, body, 10)
    file.seek(0)
    out3 = file.read().strip()
    self.assertTrue(out3 == 'cond ...')

    file = tempfile.TemporaryFile('w+')
    with jax.disable_jit():
      bm.while_loop(body, cond, 10)
      file.seek(0)
      out4 = file.read().strip()
      self.assertTrue(out4 == 'cond ...')

    file = tempfile.TemporaryFile('w+')
    with jax.disable_jit():
      jax.lax.while_loop(cond, body, -5)
    file.seek(0)
    out5 = file.read().strip()
    expect = '''
cond ...
body ...
cond ...
body ...
cond ...
body ...
cond ...
body ...
cond ...
body ...
cond ...
body ...
cond ...
    
    '''
    self.assertTrue(out5 == expect.strip())

    file = tempfile.TemporaryFile('w+')
    with jax.disable_jit():
      bm.while_loop(body, cond, -5)
    file.seek(0)
    out6 = file.read().strip()
    self.assertTrue(out5 == out6)



