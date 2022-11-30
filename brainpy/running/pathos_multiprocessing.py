# -*- coding: utf-8 -*-


"""The parallel execution of a BrainPy func on multiple CPU cores.

Specifically, these batch running functions include:

- ``cpu_ordered_parallel``: Performs a parallel ordered map.
- ``cpu_unordered_parallel``: Performs a parallel unordered map.
"""

from collections.abc import Sized
from typing import (Any, Callable, Generator, Iterable, List,
                    Union, Optional, Sequence, Dict)

from tqdm.auto import tqdm

from brainpy.errors import PackageMissingError

try:
  from pathos.helpers import cpu_count
  from pathos.multiprocessing import ProcessPool
except ModuleNotFoundError:
  cpu_count = None
  ProcessPool = None

__all__ = [
  'cpu_ordered_parallel',
  'cpu_unordered_parallel',
]


def _parallel(
    ordered: bool,
    function: Callable,
    arguments: Union[Sequence[Iterable], Dict[str, Iterable]],
    num_process: Union[int, float] = None,
    num_task: int = None,
    **tqdm_kwargs: Any
) -> Generator:
  """Perform a parallel map with a progress bar.

  Parameters
  ----------
  ordered: bool
    True for an ordered map, false for an unordered map.
  function: callable, function
    The function to apply to each element of the given Iterables.
  arguments: sequence of Iterable, dict
    One or more Iterables containing the data to be mapped.
  num_process: int, float
    Number of threads used for parallel running. If `int`, it is
    the number of threads to be used; if `float`, it is the fraction
    of total threads to be used for running.
  num_task: int
    The total number of tasks in this parallel running.
  tqdm_kwargs: Any
    The setting for the progress bar.

  Returns
  -------
  results: Iterable
      A generator which will apply the function to each element of the given Iterables
      in parallel in order with a progress bar.
  """
  if ProcessPool is None or cpu_count is None:
    raise PackageMissingError(
      '''
    Please install "pathos" package first. 
    
    >>>  pip install pathos
      '''
    )

  # Determine num_process
  if num_process is None:
    num_process = cpu_count()
  elif isinstance(num_process, int):
    pass
  elif isinstance(num_process, float):
    num_process = int(round(num_process * cpu_count()))
  else:
    raise ValueError('"num_process" must be an int or a float.')

  # arguments
  if isinstance(arguments, dict):
    keys = list(arguments.keys())
    arguments = list(arguments.values())
    run_f = lambda *args: function(**{key: arg for key, arg in zip(keys, args)})
  else:
    if not isinstance(arguments, (tuple, list)):
      raise TypeError('"arguments" must be a sequence of Iterable or a dict of Iterable. '
                      f'But we got {type(arguments)}')
    run_f = function

  # Determine length of tqdm
  lengths = [len(iterable) for iterable in arguments if isinstance(iterable, Sized)]
  num_task = num_task or (min(lengths) if lengths else None)

  # Create parallel generator
  pool = ProcessPool(nodes=num_process)
  if ordered:
    map_func = pool.imap
  else:
    map_func = pool.uimap

  # Choose tqdm variant
  for item in tqdm(map_func(run_f, *arguments), total=num_task, **tqdm_kwargs):
    yield item

  pool.clear()


def cpu_ordered_parallel(
    func: Callable,
    arguments: Union[Sequence[Iterable], Dict[str, Iterable]],
    num_process: Optional[Union[int, float]] = None,
    num_task: Optional[int] = None,
    **tqdm_kwargs: Any
) -> List[Any]:
  """Performs a parallel ordered map with a progress bar.

  Examples
  --------

  >>> import brainpy as bp
  >>> import brainpy.math as bm
  >>> import numpy as np
  >>>
  >>> def simulate(inp):
  >>>   inp = bm.as_jax(inp)
  >>>   hh = bp.neurons.HH(1)
  >>>   runner = bp.DSRunner(hh, inputs=['input', inp],
  >>>                        monitors=['V', 'spike'],
  >>>                        progress_bar=False)
  >>>   runner.run(100)
  >>>   bm.clear_buffer_memory()  # clear all cached data and functions
  >>>   return runner.mon.spike.sum()
  >>>
  >>> if __name__ == '__main__':  # This is important!
  >>>   results = bp.running.cpu_unordered_parallel(simulate, [np.arange(1, 10, 100)], num_process=10)
  >>>   print(results)

  Parameters
  ----------
  func: callable, function
    The function to apply to each element of the given Iterables.
  arguments: sequence of Iterable, dict
    One or more Iterables containing the data to be mapped.
  num_process: int, float
    Number of threads used for parallel running. If `int`, it is
    the number of threads to be used; if `float`, it is the fraction
    of total threads to be used for running.
  num_task: int
    The total number of tasks in this parallel running.
  tqdm_kwargs: Any
    The setting for the progress bar.

  Returns
  -------
  results: list
    A list which will apply the function to each element of the given tasks.
  """
  generator = _parallel(True,
                        func,
                        arguments,
                        num_process=num_process,
                        num_task=num_task,
                        **tqdm_kwargs)
  return list(generator)


def cpu_unordered_parallel(
    func: Callable,
    arguments: Union[Sequence[Iterable], Dict[str, Iterable]],
    num_process: Optional[Union[int, float]] = None,
    num_task: Optional[int] = None,
    **tqdm_kwargs: Any
) -> List[Any]:
  """Performs a parallel unordered map with a progress bar.

  Examples
  --------
  >>> import brainpy as bp
  >>> import brainpy.math as bm
  >>> import numpy as np
  >>>
  >>> def simulate(inp):
  >>>   inp = bm.as_jax(inp)
  >>>   hh = bp.neurons.HH(1)
  >>>   runner = bp.DSRunner(hh, inputs=['input', inp],
  >>>                        monitors=['V', 'spike'],
  >>>                        progress_bar=False)
  >>>   runner.run(100)
  >>>   bm.clear_buffer_memory()  # clear all cached data and functions
  >>>   return runner.mon.spike.sum()
  >>>
  >>> if __name__ == '__main__':  # This is important!
  >>>   results = bp.running.cpu_unordered_parallel(simulate, [np.arange(1, 10, 100)], num_process=10)
  >>>   print(results)

  Parameters
  ----------
  func: callable, function
    The function to apply to each element of the given Iterables.
  arguments: sequence of Iterable, dict
    One or more Iterables containing the data to be mapped.
  num_process: int, float
    Number of threads used for parallel running. If `int`, it is
    the number of threads to be used; if `float`, it is the fraction
    of total threads to be used for running.
  num_task: int
    The total number of tasks in this parallel running.
  tqdm_kwargs: Any
    The setting for the progress bar.

  Returns
  -------
  results: list
    A list which will apply the function to each element of the given tasks.
  """
  generator = _parallel(False,
                        func,
                        arguments,
                        num_process=num_process,
                        num_task=num_task,
                        **tqdm_kwargs)
  return list(generator)
