# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from datetime import datetime
from time import time

import pytest

import brainpy as bp

try:
    import pandas as pd

    df = pd.DataFrame(
        columns=['connector name', 'connect matrix size',
                 'build function', 'other parameter',
                 'time origin(ms)', 'time optimized(ms)'])
except (ImportError, ModuleNotFoundError):
    pytest.skip('No pandas installed, skip test.', allow_module_level=True)

pytest.skip('skip test.', allow_module_level=True)
# size_same = [100, 500, 2500, 12500, 25000, 37500, 50000]
# size_same = [100, 500, 2500, 12500]
size_same = [100, 500, 2500]


def get_ms(value):
    return round(value * 1000, 4)


def insert_row(connector_name, connect_matrix_size,
               build_function, other_parameter,
               time_origin_used, time_optimized_used):
    try:
        df.loc[len(df)] = [connector_name, connect_matrix_size,
                           build_function, other_parameter,
                           time_origin_used, time_optimized_used]
    except (NameError, UnboundLocalError):
        print('No pandas installed, skip test.')


def test_GaussianProb1():
    conn = bp.connect.GaussianProb(sigma=1., include_self=False, seed=123)
    for size in size_same:
        conn(pre_size=size)
        mat = conn.build_mat(isOptimized=True)
        time0 = time()
        mat1 = conn.build_mat(isOptimized=True)
        time_optimized = get_ms(time() - time0)

        mat2 = conn.build_mat(isOptimized=False)
        time0 = time()
        mat2 = conn.build_mat(isOptimized=False)
        time_origin = get_ms(time() - time0)

        assert bp.math.array_equal(mat1, mat2)
        print()
        print(f'time_optimized:{time_optimized}\ntime_origin:{time_origin}')
        insert_row('GaussianProb',
                   f'{size}x{size}',
                   'build_mat',
                   'sigma=1 / include_self=False',
                   time_origin,
                   time_optimized)


def test_GaussianProb2():
    conn = bp.connect.GaussianProb(sigma=4, seed=123)
    for size in size_same:
        conn(pre_size=size)
        mat = conn.build_mat(isOptimized=True)
        time0 = time()
        mat1 = conn.build_mat(isOptimized=True)
        time_optimized = get_ms(time() - time0)

        mat2 = conn.build_mat(isOptimized=False)
        time0 = time()
        mat2 = conn.build_mat(isOptimized=False)
        time_origin = get_ms(time() - time0)

        assert bp.math.array_equal(mat1, mat2)
        print()
        print(f'time_optimized:{time_optimized}\ntime_origin:{time_origin}')
        insert_row('GaussianProb',
                   f'{size}x{size}',
                   'build_mat',
                   'sigma=4',
                   time_origin,
                   time_optimized)


def test_GaussianProb3():
    conn = bp.connect.GaussianProb(sigma=4, periodic_boundary=True, seed=123)
    for size in size_same:
        conn(pre_size=size)
        mat = conn.build_mat(isOptimized=True)
        time0 = time()
        mat1 = conn.build_mat(isOptimized=True)
        time_optimized = get_ms(time() - time0)

        mat2 = conn.build_mat(isOptimized=False)
        time0 = time()
        mat2 = conn.build_mat(isOptimized=False)
        time_origin = get_ms(time() - time0)

        assert bp.math.array_equal(mat1, mat2)
        print()
        print(f'time_optimized:{time_optimized}\ntime_origin:{time_origin}')
        insert_row('GaussianProb',
                   f'{size}x{size}',
                   'build_mat',
                   'sigma=4 / periodic_boundary=True',
                   time_origin,
                   time_optimized)


def testGaussianProb4():
    conn = bp.connect.GaussianProb(sigma=4, periodic_boundary=True, seed=123)
    for size in size_same:
        conn(pre_size=size)
        mat = conn.build_mat(isOptimized=True)
        time0 = time()
        mat1 = conn.build_mat(isOptimized=True)
        time_optimized = get_ms(time() - time0)

        mat2 = conn.build_mat(isOptimized=False)
        time0 = time()
        mat2 = conn.build_mat(isOptimized=False)
        time_origin = get_ms(time() - time0)

        assert bp.math.array_equal(mat1, mat2)
        print()
        print(f'time_optimized:{time_optimized}\ntime_origin:{time_origin}')
        insert_row('GaussianProb',
                   f'{size}x{size}',
                   'build_mat',
                   'sigma=4 / periodic_boundary=True',
                   time_origin,
                   time_optimized)


def test_ScaleFreeBA():
    conn = bp.connect.ScaleFreeBA(m=2, seed=123)
    for size in size_same:
        conn(pre_size=size, post_size=size)
        mat = conn.build_mat(isOptimized=True)
        time0 = time()
        mat1 = conn.build_mat(isOptimized=True)
        time_optimized = get_ms(time() - time0)

        mat2 = conn.build_mat(isOptimized=False)
        time0 = time()
        mat2 = conn.build_mat(isOptimized=False)
        time_origin = get_ms(time() - time0)

        assert bp.math.array_equal(mat1, mat2)
        insert_row('ScaleFreeBA',
                   f'{size}x{size}',
                   'build_mat',
                   'm=2',
                   time_origin,
                   time_optimized)


def test_ScaleFreeBADual():
    conn = bp.connect.ScaleFreeBADual(m1=2, m2=3, p=0.4, seed=123)
    for size in size_same:
        conn(pre_size=size, post_size=size)
        mat = conn.build_mat(isOptimized=True)
        time0 = time()
        mat1 = conn.build_mat(isOptimized=True)
        time_optimized = get_ms(time() - time0)

        mat2 = conn.build_mat(isOptimized=False)
        time0 = time()
        mat2 = conn.build_mat(isOptimized=False)
        time_origin = get_ms(time() - time0)

        assert bp.math.array_equal(mat1, mat2)
        insert_row('ScaleFreeBADual',
                   f'{size}x{size}',
                   'build_mat',
                   'm1=2 / m2=3 / p=0.4',
                   time_origin,
                   time_optimized)


def test_PowerLaw():
    conn = bp.connect.PowerLaw(m=3, p=0.4, seed=123)
    for size in size_same:
        conn(pre_size=size, post_size=size)
        mat = conn.build_mat(isOptimized=True)
        time0 = time()
        mat1 = conn.build_mat(isOptimized=True)
        time_optimized = get_ms(time() - time0)

        mat2 = conn.build_mat(isOptimized=False)
        time0 = time()
        mat2 = conn.build_mat(isOptimized=False)
        time_origin = get_ms(time() - time0)

        assert bp.math.array_equal(mat1, mat2)
        insert_row('PowerLaw',
                   f'{size}x{size}',
                   'build_mat',
                   'm=3 / p=0.4',
                   time_origin,
                   time_optimized)


def test_ProbDist():
    conn = bp.connect.ProbDist(dist=1, prob=0.5, pre_ratio=0.3, seed=123, include_self=True)
    # for size in [1000, (100, 20), (4, 20, 20), (4, 3, 8, 5)]:
    for size in [10000]:
        conn(pre_size=size, post_size=size)
        pre_ids1, post_ids1 = conn.build_coo(isOptimized=True)
        time0 = time()
        pre_ids1, post_ids1 = conn.build_coo(isOptimized=True)
        time_optimized = get_ms(time() - time0)

        pre_ids2, post_ids2 = conn.build_coo(isOptimized=False)
        time0 = time()
        pre_ids2, post_ids2 = conn.build_coo(isOptimized=False)
        time_origin = get_ms(time() - time0)

        # assert (bp.math.array_equal(pre_ids1, pre_ids2) and bp.math.array_equal(post_ids1, post_ids2))
        print()
        print(f'time origin: {time_origin}\ntime optimized: {time_optimized}')
        insert_row('ProbDist',
                   {size},
                   'build_coo',
                   'dist=1 / prob=0.5 / pre_ratio=0.3 / include_self=True',
                   time_origin,
                   time_optimized)


def test_save():
    try:
        df.to_csv('opt_time_compare' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv',
                  index=False)
    except (NameError, UnboundLocalError):
        print('No pandas installed, skip test.')
