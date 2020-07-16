# -*- coding: utf-8 -*-

import multiprocessing

__all__ = [
    'process_pool',
    'process_pool_lock',
]


def process_pool(func, all_net_params, nb_process):
    print('{} jobs total.'.format(len(all_net_params)))
    pool = multiprocessing.Pool(processes=nb_process)
    results = []
    for net_params in all_net_params:
        results.append(pool.apply_async(func, args=net_params))
    pool.close()
    pool.join()
    return results


def process_pool_lock(func, all_net_params, nb_process):
    print('{} jobs total.'.format(len(all_net_params)))
    pool = multiprocessing.Pool(processes=nb_process)
    m = multiprocessing.Manager()
    lock = m.Lock()
    results = []
    for net_params in all_net_params:
        results.append(pool.apply_async(func, args=tuple(net_params) + (lock,)))
    pool.close()
    pool.join()
    return results
