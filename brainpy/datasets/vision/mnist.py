# -*- coding: utf-8 -*-


from brainpy.errors import NoLongerSupportError

__all__ = [
  'MNIST',
  'FashionMNIST',
  'KMNIST',
  'EMNIST',
  'QMNIST',
]


class MNIST(object):
  def __init__(self, *args, **kwargs):
    raise NoLongerSupportError('''
    Install ``brainpy-datasets`` first. Then use 

    >>> import brainpy_datasets as bd
    >>> bd.vision.MNIST
                ''')


class FashionMNIST(object):
  def __init__(self, *args, **kwargs):
    raise NoLongerSupportError('''
    Install ``brainpy-datasets`` first. Then use 

    >>> import brainpy_datasets as bd
    >>> bd.vision.FashionMNIST
                ''')


class KMNIST(object):
  def __init__(self, *args, **kwargs):
    raise NoLongerSupportError('''
    Install ``brainpy-datasets`` first. Then use 

    >>> import brainpy_datasets as bd
    >>> bd.vision.KMNIST
                ''')


class EMNIST(object):
  def __init__(self, *args, **kwargs):
    raise NoLongerSupportError('''
    Install ``brainpy-datasets`` first. Then use 

    >>> import brainpy_datasets as bd
    >>> bd.vision.EMNIST
                ''')


class QMNIST(object):
  def __init__(self, *args, **kwargs):
    raise NoLongerSupportError('''
    Install ``brainpy-datasets`` first. Then use 

    >>> import brainpy_datasets as bd
    >>> bd.vision.QMNIST
                ''')
