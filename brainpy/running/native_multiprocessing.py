# -*- coding: utf-8 -*-

from typing import Union, Sequence, Dict
import multiprocessing

__all__ = [
  'process_pool',
  'process_pool_lock',
]


def process_pool(func: callable,
                 all_params: Union[Sequence, Dict],
                 num_process: int):
  """Run multiple models in multi-processes.

  .. Note::
     This multiprocessing function should be called within a `if __main__ == '__main__':` syntax.

  Parameters
  ----------
  func : callable
      The function to run model.
  all_params : list, tuple, dict
      The parameters of the function arguments.
      The parameters for each process can be a tuple, or a dictionary.
  num_process : int
      The number of the processes.

  Returns
  -------
  results : list
      Process results.
  """
  print('{} jobs total.'.format(len(all_params)))
  pool = multiprocessing.Pool(processes=num_process)
  results = []
  for params in all_params:
    if isinstance(params, (list, tuple)):
      results.append(pool.apply_async(func, args=tuple(params)))
    elif isinstance(params, dict):
      results.append(pool.apply_async(func, kwds=params))
    else:
      raise ValueError('Unknown parameter type: ', type(params))
  pool.close()
  pool.join()
  return [r.get() for r in results]


def process_pool_lock(func: callable,
                      all_params: Union[Sequence, Dict],
                      num_process: int):
  """Run multiple models in multi-processes with lock.

  Sometimes, you want to synchronize the processes. For example,
  if you want to write something in a document, you cannot let
  multiprocess simultaneously open this same file. So, you need
  add a `lock` argument in your defined `func`:

  .. code-block:: python

      def some_func(..., lock, ...):
          ... do something ..

          lock.acquire()
          ... something cannot simultaneously do by multi-process ..
          lock.release()

  In such case, you can use `process_pool_lock()` to run your model.

  .. Note::
     This multiprocessing function should be called within a `if __main__ == '__main__':` syntax.

  Parameters
  ----------
  func: callable
      The function to run model.
  all_params : list, tuple, dict
      The parameters of the function arguments.
  num_process : int
      The number of the processes.

  Returns
  -------
  results : list
      Process results.
  """
  print('{} jobs total.'.format(len(all_params)))
  pool = multiprocessing.Pool(processes=num_process)
  m = multiprocessing.Manager()
  lock = m.Lock()
  results = []
  for net_params in all_params:
    if isinstance(net_params, (list, tuple)):
      results.append(pool.apply_async(func, args=tuple(net_params) + (lock,)))
    elif isinstance(net_params, dict):
      net_params.update(lock=lock)
      results.append(pool.apply_async(func, kwds=net_params))
    else:
      raise ValueError('Unknown parameter type: ', type(net_params))
  pool.close()
  pool.join()
  return [r.get() for r in results]
