# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
import argparse


class SNN(bp.DynamicalSystem):
  def __init__(self, tau):
    super().__init__()

    self.layer = bp.Sequential(
      bp.layers.Flatten(),
      bp.layers.Dense(28 * 28, 10, b_initializer=None),
      bp.neurons.LIF(tau=tau, spike_fun=bm.surrogate.arctan),
    )

  def update(self, p, x):
    return self.layer(p, x)

parser = argparse.ArgumentParser(description='LIF MNIST Training')
parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
parser.add_argument('-device', default='cpu', help='device')
parser.add_argument('-b', default=64, type=int, help='batch size')
parser.add_argument('-epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-data-dir', type=str, default=r'D:\data', help='root dir of MNIST dataset')
parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam',
                    help='use which optimizer. SGD or Adam')
parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

args = parser.parse_args()
print(args)

