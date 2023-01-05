# -*- coding: utf-8 -*-

import unittest
import brainpy.math as bm


import jax.numpy
import matplotlib.pyplot as plt
from absl.testing import parameterized

from brainpy._src.optimizers import scheduler


class TestMultiStepLR(parameterized.TestCase):

  @parameterized.named_parameters(
    {'testcase_name': f'last_epoch={last_epoch}',
     'last_epoch': last_epoch}
    for last_epoch in [-1, 0, 5, 10]
  )
  def test2(self, last_epoch):
    scheduler1 = scheduler.MultiStepLR(0.1, [10, 20], gamma=0.1, last_epoch=last_epoch)
    scheduler2 = scheduler.MultiStepLR(0.1, [10, 20], gamma=0.1, last_epoch=last_epoch)

    for i in range(1, 25):
      lr1 = scheduler1(i + last_epoch)
      lr2 = scheduler2()
      scheduler2.step_epoch()
      print(f'{scheduler2.last_epoch}, {lr1:.4f}, {lr2:.4f}')
      self.assertTrue(lr1 == lr2)


class TestStepLR(parameterized.TestCase):

  @parameterized.named_parameters(
    {'testcase_name': f'last_epoch={last_epoch}',
     'last_epoch': last_epoch}
    for last_epoch in [-1, 0, 5, 10]
  )
  def test1(self, last_epoch):
    scheduler1 = scheduler.StepLR(0.1, 10, gamma=0.1, last_epoch=last_epoch)
    scheduler2 = scheduler.StepLR(0.1, 10, gamma=0.1, last_epoch=last_epoch)

    for i in range(1, 25):
      lr1 = scheduler1(i + last_epoch)
      lr2 = scheduler2()
      scheduler2.step_epoch()
      print(f'{scheduler2.last_epoch}, {lr1:.4f}, {lr2:.4f}')
      self.assertTrue(lr1 == lr2)


class TestCosineAnnealingLR(unittest.TestCase):
  def test1(self):
    max_epoch = 50
    iters = 200
    sch = scheduler.CosineAnnealingLR(0.1, T_max=5, eta_min=0, last_epoch=-1)
    all_lr1 = [[], []]
    all_lr2 = [[], []]
    for epoch in range(max_epoch):
      for batch in range(iters):
        all_lr1[0].append(epoch + batch / iters)
        all_lr1[1].append(sch())
        sch.step_epoch()
      all_lr2[0].append(epoch)
      all_lr2[1].append(sch())
      sch.step_epoch()
    plt.subplot(211)
    plt.plot(jax.numpy.asarray(all_lr1[0]), jax.numpy.asarray(all_lr1[1]))
    plt.subplot(212)
    plt.plot(jax.numpy.asarray(all_lr2[0]), jax.numpy.asarray(all_lr2[1]))
    plt.show()
    plt.close()


class TestCosineAnnealingWarmRestarts(unittest.TestCase):
  def test1(self):
    max_epoch = 50
    iters = 200
    sch = scheduler.CosineAnnealingWarmRestarts(0.1,
                                                iters,
                                                T_0=5,
                                                T_mult=1,
                                                last_call=-1)
    all_lr1 = []
    all_lr2 = []
    for epoch in range(max_epoch):
      for batch in range(iters):
        all_lr1.append(sch())
        sch.step_call()
      all_lr2.append(sch())
      sch.step_epoch()
    plt.subplot(211)
    plt.plot(jax.numpy.asarray(all_lr1))
    plt.subplot(212)
    plt.plot(jax.numpy.asarray(all_lr2))
    plt.show()
    plt.close()


