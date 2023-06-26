# -*- coding: utf-8 -*-

import pytest

import unittest

import brainpy as bp

from time import time


def test_gaussian_prob1():
    conn = bp.connect.GaussianProb(sigma=1., include_self=False, seed=123)(pre_size=100)

    mat = conn.build_mat(isOptimized=True)
    time0 = time()
    mat1 = conn.build_mat(isOptimized=True)
    time_optimized = time() - time0

    time0 = time()
    mat2 = conn.build_mat(isOptimized=False)
    time_origin = time() - time0

    assert bp.math.array_equal(mat1, mat2)
    print()
    print(f'time_optimized:{time_optimized}\ntime_origin:{time_origin}')


def test_gaussian_prob2():
    conn = bp.connect.GaussianProb(sigma=4, seed=123)(pre_size=(10, 10))
    mat = conn.build_mat(isOptimized=True)
    time0 = time()
    mat1 = conn.build_mat(isOptimized=True)
    time_optimized = time() - time0

    time0 = time()
    mat2 = conn.build_mat(isOptimized=False)
    time_origin = time() - time0

    assert bp.math.array_equal(mat1, mat2)
    print()
    print(f'time_optimized:{time_optimized}\ntime_origin:{time_origin}')


def test_gaussian_prob3():
    conn = bp.connect.GaussianProb(sigma=4, periodic_boundary=True, seed=123)(pre_size=(10, 10))
    mat = conn.build_mat(isOptimized=True)
    time0 = time()
    mat1 = conn.build_mat(isOptimized=True)
    time_optimized = time() - time0

    time0 = time()
    mat2 = conn.build_mat(isOptimized=False)
    time_origin = time() - time0

    assert bp.math.array_equal(mat1, mat2)
    print()
    print(f'time_optimized:{time_optimized}\ntime_origin:{time_origin}')


def test_gaussian_prob4():
    conn = bp.connect.GaussianProb(sigma=4, periodic_boundary=True, seed=123)(pre_size=(10, 10, 10))
    mat = conn.build_mat(isOptimized=True)
    time0 = time()
    mat1 = conn.build_mat(isOptimized=True)
    time_optimized = time() - time0

    time0 = time()
    mat2 = conn.build_mat(isOptimized=False)
    time_origin = time() - time0

    assert bp.math.array_equal(mat1, mat2)
    print()
    print(f'time_optimized:{time_optimized}\ntime_origin:{time_origin}')
