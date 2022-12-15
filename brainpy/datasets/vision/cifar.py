from brainpy.errors import NoLongerSupportError

__all__ = [
  'CIFAR10',
  'CIFAR100'
]


class CIFAR10(object):
  def __init__(self, *args, **kwargs) -> None:
    raise NoLongerSupportError('''
Install ``brainpy-datasets`` first. Then use 

>>> import brainpy_datasets as bd
>>> bd.vision.CIFAR10
        ''')


class CIFAR100(object):
  def __init__(self, *args, **kwargs):
    raise NoLongerSupportError('''
    Install ``brainpy-datasets`` first. Then use 

    >>> import brainpy_datasets as bd
    >>> bd.vision.CIFAR100
                ''')
