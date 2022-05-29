# -*- coding: utf-8 -*-

import inspect
from typing import Union, Callable, Optional, Dict

from brainpy.train.algorithms import OfflineAlgorithm, OnlineAlgorithm
from brainpy.dyn.base import DynamicalSystem
from brainpy.types import Tensor

__all__ = [
  'TrainingSystem', 'Sequential',
]


def not_implemented(fun: Callable) -> Callable:
  """Marks the given module method is not implemented.

  Methods wrapped in @not_implemented can define submodules directly within the method.

  For instance::

    @not_implemented
    init_fb(self):
      ...

    @not_implemented
    def feedback(self):
      ...
  """
  fun.not_implemented = True
  return fun


class TrainingSystem(DynamicalSystem):
  """Base class for training system in BrainPy.

  """

  '''Online fitting method.'''
  online_fit_by: Optional[OnlineAlgorithm]

  '''Offline fitting method.'''
  offline_fit_by: Optional[OfflineAlgorithm]

  def __init__(self, name: str = None, trainable: bool = False):
    super(TrainingSystem, self).__init__(name=name)
    self._trainable = trainable
    self.online_fit_by = None
    self.offline_fit_by = None
    self.fit_record = dict()

  @property
  def trainable(self):
    return self._trainable

  def __repr__(self):
    return f"{type(self).__name__}(name={self.name}, trainable={self.trainable})"

  def __call__(self, *args, **kwargs) -> Tensor:
    """The main computation function of a Node.

    Returns
    -------
    Tensor
      A output tensor value, or a dict of output tensors.
    """
    return self.forward(*args, **kwargs)

  @not_implemented
  def update(self, t, dt, x, shared_args=None) -> Tensor:
    return self.forward(x, shared_args)

  def forward(self, x, shared_args=None) -> Tensor:
    raise NotImplementedError('Subclass should implement "forward()" function '
                              'when "update()" function is not customized.')

  def reset(self, batch_size=1):
    for node in self.nodes(level=1, include_self=False).unique().subset(TrainingSystem).values():
      node.reset(batch_size=batch_size)

  def reset_batch_state(self, batch_size=1):
    for node in self.nodes(level=1, include_self=False).unique().subset(TrainingSystem).values():
      node.reset_batch_state(batch_size=batch_size)

  @not_implemented
  def online_init(self):
    raise NotImplementedError('Subclass must implement online_init() function when using '
                              'OnlineTrainer.')

  @not_implemented
  def offline_init(self):
    raise NotImplementedError('Subclass must implement offline_init() function when using '
                              'OfflineTrainer.')

  @not_implemented
  def online_fit(self,
                 target: Tensor,
                 fit_record: Dict[str, Tensor]):
    raise NotImplementedError('Subclass must implement online_fit() function when using '
                              'OnlineTrainer.')

  @not_implemented
  def offline_fit(self,
                  target: Tensor,
                  fit_record: Dict[str, Tensor],
                  shared_args: Dict = None):
    raise NotImplementedError('Subclass must implement offline_fit() function when using '
                              'OfflineTrainer.')



class Sequential(TrainingSystem):
  def __init__(self, *modules, name: str = None, **kw_modules):
    super(Sequential, self).__init__(name=name, trainable=False)

    # add sub-components
    for module in modules:
      if isinstance(module, TrainingSystem):
        self.implicit_nodes[module.name] = module
      elif isinstance(module, (list, tuple)):
        for m in module:
          if not isinstance(m, TrainingSystem):
            raise ValueError(f'Should be instance of {TrainingSystem.__name__}. '
                             f'But we got {type(m)}')
          self.implicit_nodes[m.name] = module
      elif isinstance(module, dict):
        for k, v in module.items():
          if not isinstance(v, TrainingSystem):
            raise ValueError(f'Should be instance of {TrainingSystem.__name__}. '
                             f'But we got {type(v)}')
          self.implicit_nodes[k] = v
      else:
        raise ValueError(f'Cannot parse sub-systems. They should be {TrainingSystem.__name__} '
                         f'or a list/tuple/dict of  {TrainingSystem.__name__}.')
    for k, v in kw_modules.items():
      if not isinstance(v, TrainingSystem):
        raise ValueError(f'Should be instance of {TrainingSystem.__name__}. '
                         f'But we got {type(v)}')
      self.implicit_nodes[k] = v

  def __getattr__(self, item):
    """Wrap the dot access ('self.'). """
    child_ds = super(Sequential, self).__getattribute__('implicit_nodes')
    if item in child_ds:
      return child_ds[item]
    else:
      return super(Sequential, self).__getattribute__(item)

  def __getitem__(self, key: Union[int, slice]):
    if isinstance(key, str):
      if key not in self.implicit_nodes:
        raise KeyError(f'Does not find a component named {key} in\n {str(self)}')
      return self.implicit_nodes[key]
    elif isinstance(key, slice):
      keys = tuple(self.implicit_nodes.keys())[key]
      components = tuple(self.implicit_nodes.values())[key]
      return Sequential(dict(zip(keys, components)))
    elif isinstance(key, int):
      return self.implicit_nodes.values()[key]
    elif isinstance(key, (tuple, list)):
      for i in key:
        if isinstance(i, int):
          raise KeyError(f'We excepted a tuple/list of int, but we got {type(i)}')
      keys = tuple(self.implicit_nodes.keys())[key]
      components = tuple(self.implicit_nodes.values())[key]
      return Sequential(dict(zip(keys, components)))
    else:
      raise KeyError(f'Unknown type of key: {type(key)}')

  def __repr__(self):
    def f(x):
      if not isinstance(x, TrainingSystem) and callable(x):
        signature = inspect.signature(x)
        args = [f'{k}={v.default}' for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty]
        args = ', '.join(args)
        while not hasattr(x, '__name__'):
          if not hasattr(x, 'func'):
            break
          x = x.func  # Handle functools.partial
        if not hasattr(x, '__name__') and hasattr(x, '__class__'):
          return x.__class__.__name__
        if args:
          return f'{x.__name__}(*, {args})'
        return x.__name__
      else:
        x = repr(x).split('\n')
        x = [x[0]] + ['  ' + y for y in x[1:]]
        return '\n'.join(x)

    entries = '\n'.join(f'  [{i}] {f(x)}' for i, x in enumerate(self))
    return f'{self.__class__.__name__}(\n{entries}\n)'

  def forward(self, x, shared_args=None) -> Tensor:
    for node in self.implicit_nodes.values():
      x = node.forward(x, shared_args=shared_args)
    return x
