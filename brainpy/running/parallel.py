# -*- coding: utf-8 -*-

import multiprocessing

__all__ = [
  'process_pool',
  'process_pool_lock',
]


def process_pool(func, all_net_params, nb_process):
  """Run multiple models in multi-processes.

  Parameters
  ----------
  func : callable
      The function to run model.
  all_net_params : a_list, tuple
      The parameters of the function arguments.
      The parameters for each process can be a tuple, or a dictionary.
  nb_process : int
      The number of the processes.

  Returns
  -------
  results : list
      Process results.
  """
  print('{} jobs total.'.format(len(all_net_params)))
  pool = multiprocessing.Pool(processes=nb_process)
  results = []
  for net_params in all_net_params:
    if isinstance(net_params, (list, tuple)):
      results.append(pool.apply_async(func, args=tuple(net_params)))
    elif isinstance(net_params, dict):
      results.append(pool.apply_async(func, kwds=net_params))
    else:
      raise ValueError('Unknown parameter type: ', type(net_params))
  pool.close()
  pool.join()
  return results


def process_pool_lock(func, all_net_params, nb_process):
  """Run multiple models in multi-processes with lock.

  Sometimes, you want to synchronize the processes. For example,
  if you want to write something in a document, you cannot let
  multi-process simultaneously open this same file. So, you need
  add a `lock` argument in your defined `func`:

  .. code-block:: python

      def some_func(..., lock, ...):
          ... do something ..

          lock.acquire()
          ... something cannot simultaneously do by multi-process ..
          lock.release()

  In such case, you can use `process_pool_lock()` to run your model.

  Parameters
  ----------
  func : callable
      The function to run model.
  all_net_params : a_list, tuple
      The parameters of the function arguments.
  nb_process : int
      The number of the processes.

  Returns
  -------
  results : list
      Process results.
  """
  print('{} jobs total.'.format(len(all_net_params)))
  pool = multiprocessing.Pool(processes=nb_process)
  m = multiprocessing.Manager()
  lock = m.Lock()
  results = []
  for net_params in all_net_params:
    if isinstance(net_params, (list, tuple)):
      results.append(pool.apply_async(func, args=tuple(net_params) + (lock,)))
    elif isinstance(net_params, dict):
      net_params.update(lock=lock)
      results.append(pool.apply_async(func, kwds=net_params))
    else:
      raise ValueError('Unknown parameter type: ', type(net_params))
  pool.close()
  pool.join()
  return results
