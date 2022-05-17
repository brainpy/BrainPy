import unittest

import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as nr

import brainpy.math as bm
import brainpy.math.random as br


class TestRandom(unittest.TestCase):
  def test_seed(self):
    test_seed = 299
    br.seed(test_seed)
    a = br.rand(3)
    br.seed(test_seed)
    b = br.rand(3)
    self.assertTrue(bm.array_equal(a, b))

  def test_rand(self):
    a = br.rand(3, 2)
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a < 1).all())

  def test_randint1(self):
    a = br.randint(5)
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(0 <= a < 5)

  def test_randint2(self):
    a = br.randint(2, 6, size=(4, 3))
    self.assertTupleEqual(a.shape, (4, 3))
    self.assertTrue((a >= 2).all() and (a < 6).all())

  def test_randint3(self):
    a = br.randint([1, 2, 3], [10, 7, 8])
    self.assertTupleEqual(a.shape, (3,))
    self.assertTrue((a - bm.array([1, 2, 3]) >= 0).all()
                    and (-a + bm.array([10, 7, 8]) > 0).all())

  def test_randint4(self):
    a = br.randint([1, 2, 3], [10, 7, 8], size=(2, 3))
    self.assertTupleEqual(a.shape, (2, 3))

  def test_randn(self):
    a = br.randn(3, 2)
    self.assertTupleEqual(a.shape, (3, 2))

  def test_random1(self):
    a = br.random()
    self.assertIsInstance(a, bm.jaxarray.JaxArray)
    self.assertTrue(0. <= a < 1)

  def test_random2(self):
    a = br.random(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a < 1).all())

  def test_random_sample(self):
    a = br.random_sample(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a < 1).all())

  def test_choice1(self):
    a = bm.random.choice(5)
    self.assertTupleEqual(jnp.shape(a), ())
    self.assertTrue(0 <= a < 5)

  def test_choice2(self):
    a = bm.random.choice(5, 3, p=[0.1, 0.4, 0.2, 0., 0.3])
    self.assertTupleEqual(a.shape, (3,))
    self.assertTrue((a >= 0).all() and (a < 5).all())

  def test_choice3(self):
    a = bm.random.choice(bm.arange(2, 20), size=(4, 3), replace=False)
    self.assertTupleEqual(a.shape, (4, 3))
    self.assertTrue((a >= 2).all() and (a < 20).all())
    self.assertEqual(len(bm.unique(a)), 12)

  def test_permutation1(self):
    a = bm.random.permutation(10)
    self.assertTupleEqual(a.shape, (10,))
    self.assertEqual(len(bm.unique(a)), 10)

  def test_permutation2(self):
    a = bm.random.permutation(bm.arange(10))
    self.assertTupleEqual(a.shape, (10,))
    self.assertEqual(len(bm.unique(a)), 10)

  def test_shuffle1(self):
    a = bm.arange(10)
    bm.random.shuffle(a)
    self.assertTupleEqual(a.shape, (10,))
    self.assertEqual(len(bm.unique(a)), 10)

  def test_shuffle2(self):
    a = bm.arange(12).reshape(4, 3)
    bm.random.shuffle(a, axis=1)
    self.assertTupleEqual(a.shape, (4, 3))
    self.assertEqual(len(bm.unique(a)), 12)

    # test that a is only shuffled along axis 1
    uni = bm.unique(bm.diff(a, axis=0))
    self.assertEqual(uni, bm.JaxArray([3]))

  def test_beta1(self):
    a = bm.random.beta(2, 2)
    self.assertTupleEqual(a.shape, ())

  def test_beta2(self):
    a = bm.random.beta([2, 2, 3], 2, size=(3,))
    self.assertTupleEqual(a.shape, (3,))

  def test_exponential1(self):
    a = bm.random.exponential(10., size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_exponential2(self):
    a = bm.random.exponential([1., 2., 5.])
    self.assertTupleEqual(a.shape, (3,))

  def test_gamma(self):
    a = bm.random.gamma(2, 10., size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_gumbel(self):
    a = bm.random.gumbel(0., 2., size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_laplace(self):
    a = bm.random.laplace(0., 2., size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_logistic(self):
    a = bm.random.logistic(0., 2., size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_normal1(self):
    a = bm.random.normal()
    self.assertTupleEqual(a.shape, ())

  def test_normal2(self):
    a = bm.random.normal(loc=[0., 2., 4.], scale=[1., 2., 3.])
    self.assertTupleEqual(a.shape, (3,))

  def test_normal3(self):
    a = bm.random.normal(loc=[0., 2., 4.], scale=[[1., 2., 3.], [1., 1., 1.]])
    print(a)
    self.assertTupleEqual(a.shape, (2, 3))

  def test_pareto(self):
    a = bm.random.pareto([1, 2, 2])
    self.assertTupleEqual(a.shape, (3,))

  def test_poisson(self):
    a = bm.random.poisson([1., 2., 2.], size=3)
    self.assertTupleEqual(a.shape, (3,))

  def test_standard_cauchy(self):
    a = bm.random.standard_cauchy(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))

  def test_standard_exponential(self):
    a = bm.random.standard_exponential(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))

  def test_standard_gamma(self):
    a = bm.random.standard_gamma(shape=[1, 2, 4], size=3)
    self.assertTupleEqual(a.shape, (3,))

  def test_standard_normal(self):
    a = bm.random.standard_normal(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))

  def test_standard_t(self):
    a = bm.random.standard_t(df=[1, 2, 4], size=3)
    self.assertTupleEqual(a.shape, (3,))

  def test_standard_uniform1(self):
    a = bm.random.uniform()
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(0 <= a < 1)

  def test_uniform2(self):
    a = bm.random.uniform(low=[-1., 5., 2.], high=[2., 6., 10.], size=3)
    self.assertTupleEqual(a.shape, (3,))
    self.assertTrue((a - bm.array([-1., 5., 2.]) >= 0).all()
                    and (-a + bm.array([2., 6., 10.]) > 0).all())

  def test_uniform3(self):
    a = bm.random.uniform(low=-1., high=[2., 6., 10.], size=(2, 3))
    self.assertTupleEqual(a.shape, (2, 3))

  def test_uniform4(self):
    a = bm.random.uniform(low=[-1., 5., 2.], high=[[2., 6., 10.], [10., 10., 10.]])
    self.assertTupleEqual(a.shape, (2, 3))

  def test_truncated_normal1(self):
    a = bm.random.truncated_normal(-1., 1.)
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(-1. <= a <= 1.)

  def test_truncated_normal2(self):
    a = bm.random.truncated_normal(-1., [1., 2., 1.], size=(4, 3))
    self.assertTupleEqual(a.shape, (4, 3))

  def test_truncated_normal3(self):
    a = bm.random.truncated_normal([-1., 0., 1.], [[2., 2., 4.], [2., 2., 4.]])
    self.assertTupleEqual(a.shape, (2, 3))
    self.assertTrue((a - bm.array([-1., 0., 1.]) >= 0.).all()
                    and (- a + bm.array([2., 2., 4.]) >= 0.).all())

  def test_bernoulli1(self):
    a = bm.random.bernoulli()
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(a == 0 or a == 1)

  def test_bernoulli2(self):
    a = bm.random.bernoulli([0.5, 0.6, 0.8])
    self.assertTupleEqual(a.shape, (3,))
    self.assertTrue(bm.logical_xor(a == 1, a == 0).all())

  def test_bernoulli3(self):
    a = bm.random.bernoulli([0.5, 0.6], size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue(bm.logical_xor(a == 1, a == 0).all())

  def test_lognormal1(self):
    a = bm.random.lognormal()
    self.assertTupleEqual(a.shape, ())

  def test_lognormal2(self):
    a = bm.random.lognormal(sigma=[2., 1.], size=[3, 2])
    self.assertTupleEqual(a.shape, (3, 2))

  def test_lognormal3(self):
    a = bm.random.lognormal([2., 0.], [[2., 1.], [3., 1.2]])
    self.assertTupleEqual(a.shape, (2, 2))

  def test_binomial1(self):
    a = bm.random.binomial(5, 0.5)
    b = np.random.binomial(5, 0.5)
    print(a)
    print(b)
    self.assertIsInstance(a, bm.JaxArray)
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(a.dtype, int)

  def test_binomial2(self):
    a = bm.random.binomial(5, 0.5, size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a <= 5).all())

  def test_binomial3(self):
    a = bm.random.binomial(n=bm.asarray([2, 3, 4]), p=bm.asarray([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]))
    self.assertTupleEqual(a.shape, (2, 3))

  def test_chisquare1(self):
    a = bm.random.chisquare(3)
    self.assertIsInstance(a, bm.JaxArray)
    self.assertTupleEqual(a.shape, ())
    self.assertTrue(a.dtype, float)

  def test_chisquare2(self):
    with self.assertRaises(NotImplementedError):
      a = bm.random.chisquare(df=[2, 3, 4])

  def test_chisquare3(self):
    a = bm.random.chisquare(df=2, size=100)
    self.assertTupleEqual(a.shape, (100,))

  def test_chisquare4(self):
    a = bm.random.chisquare(df=2, size=(100, 10))
    self.assertTupleEqual(a.shape, (100, 10))

  def test_dirichlet1(self):
    a = bm.random.dirichlet((10, 5, 3))
    self.assertTupleEqual(a.shape, (3,))

  def test_dirichlet2(self):
    a = bm.random.dirichlet((10, 5, 3), 20)
    self.assertTupleEqual(a.shape, (20, 3))

  def test_f(self):
    a = bm.random.f(1., 48., 100)
    self.assertTupleEqual(a.shape, (100,))

  def test_geometric(self):
    a = bm.random.geometric([0.7, 0.5, 0.2])
    self.assertTupleEqual(a.shape, (3,))

  def test_hypergeometric1(self):
    a = bm.random.hypergeometric(10, 10, 10, 20)
    self.assertTupleEqual(a.shape, (20,))

  def test_hypergeometric2(self):
    a = bm.random.hypergeometric(8, [10, 4], [[5, 2], [5, 5]])
    self.assertTupleEqual(a.shape, (2, 2))

  def test_hypergeometric3(self):
    a = bm.random.hypergeometric(8, [10, 4], [[5, 2], [5, 5]], size=(3, 2, 2))
    self.assertTupleEqual(a.shape, (3, 2, 2))

  def test_logseries(self):
    a = bm.random.logseries([0.7, 0.5, 0.2], size=[4, 3])
    self.assertTupleEqual(a.shape, (4, 3))

  def test_multinominal1(self):
    a = np.random.multinomial(100, (0.5, 0.2, 0.3), size=[4, 2])
    print(a, a.shape)
    b = bm.random.multinomial(100, (0.5, 0.2, 0.3), size=[4, 2])
    print(b, b.shape)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (4, 2, 3))

  def test_multinominal2(self):
    a = bm.random.multinomial(100, (0.5, 0.2, 0.3))
    self.assertTupleEqual(a.shape, (3,))
    self.assertTrue(a.sum() == 100)

  def test_multinominal3(self):
    with self.assertRaises(ValueError):
      a = bm.random.multinomial(100, (0.5, 0.6, 0.3))
    with self.assertRaises(ValueError):
      f = jax.jit(bm.random.multinomial, static_argnums=2)
      a = f(100, (0.5, 0.6, 0.3), 2)

  def test_multivariate_normal1(self):
    # self.skipTest('Windows jaxlib error')
    a = np.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], size=3)
    b = bm.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], size=3)
    print('test_multivariate_normal1')
    print(a)
    print(b)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(a.shape, (3, 2))

  def test_multivariate_normal2(self):
    a = np.random.multivariate_normal([1, 2], [[1, 3], [3, 1]])
    b = bm.random.multivariate_normal([1, 2], [[1, 3], [3, 1]], method='svd')
    print(a)
    print(b)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(a.shape, (2,))

  def test_negative_binomial(self):
    a = np.random.negative_binomial([3., 10.], 0.5)
    b = bm.random.negative_binomial([3., 10.], 0.5)
    print(a)
    print(b)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (2,))

  def test_negative_binomial2(self):
    a = np.random.negative_binomial(3., 0.5, 10)
    b = bm.random.negative_binomial(3., 0.5, 10)
    print(a)
    print(b)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (10,))

  def test_noncentral_chisquare(self):
    a = np.random.noncentral_chisquare(3, [3., 2.], (4, 2))
    b = bm.random.noncentral_chisquare(3, [3., 2.], (4, 2))
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (4, 2))

  def test_noncentral_chisquare2(self):
    a = bm.random.noncentral_chisquare(3, [3., 2.])
    self.assertTupleEqual(a.shape, (2,))

  def test_noncentral_f(self):
    a = bm.random.noncentral_f(3, 20, 3., 100)
    self.assertTupleEqual(a.shape, (100,))

  def test_power(self):
    a = np.random.power(2, (4, 2))
    b = bm.random.power(2, (4, 2))
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (4, 2))

  def test_rayleigh(self):
    a = bm.random.power(2., (4, 2))
    self.assertTupleEqual(a.shape, (4, 2))

  def test_triangular(self):
    a = bm.random.triangular((2, 2))
    self.assertTupleEqual(a.shape, (2, 2))

  def test_vonmises(self):
    a = np.random.vonmises(2., 2.)
    b = bm.random.vonmises(2., 2.)
    print(a, b)
    self.assertTupleEqual(np.shape(a), b.shape)
    self.assertTupleEqual(b.shape, ())

  def test_vonmises2(self):
    a = np.random.vonmises(2., 2., 10)
    b = bm.random.vonmises(2., 2., 10)
    print(a, b)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (10,))

  def test_wald(self):
    a = np.random.wald([2., 0.5], 2.)
    b = bm.random.wald([2., 0.5], 2.)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (2,))

  def test_wald2(self):
    a = np.random.wald(2., 2., 100)
    b = bm.random.wald(2., 2., 100)
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (100,))

  def test_weibull(self):
    a = bm.random.weibull(2., (4, 2))
    self.assertTupleEqual(a.shape, (4, 2))

  def test_weibull2(self):
    a = bm.random.weibull(2., )
    self.assertTupleEqual(a.shape, ())

  def test_weibull3(self):
    a = bm.random.weibull([2., 3.], )
    self.assertTupleEqual(a.shape, (2,))

  def test_weibull_min(self):
    a = bm.random.weibull_min(2., 2., (4, 2))
    self.assertTupleEqual(a.shape, (4, 2))

  def test_weibull_min2(self):
    a = bm.random.weibull_min(2., 2.)
    self.assertTupleEqual(a.shape, ())

  def test_weibull_min3(self):
    a = bm.random.weibull_min([2., 3.], 2.)
    self.assertTupleEqual(a.shape, (2,))

  def test_zipf(self):
    a = bm.random.zipf(2., (4, 2))
    self.assertTupleEqual(a.shape, (4, 2))

  def test_zipf2(self):
    a = np.random.zipf([1.1, 2.])
    b = bm.random.zipf([1.1, 2.])
    self.assertTupleEqual(a.shape, b.shape)
    self.assertTupleEqual(b.shape, (2,))

  def test_maxwell(self):
    a = bm.random.maxwell(10)
    self.assertTupleEqual(a.shape, (10,))

  def test_maxwell2(self):
    a = bm.random.maxwell()
    self.assertTupleEqual(a.shape, ())

  def test_t(self):
    a = bm.random.t(1., size=10)
    self.assertTupleEqual(a.shape, (10,))

  def test_t2(self):
    a = bm.random.t([1., 2.], size=None)
    self.assertTupleEqual(a.shape, (2,))
