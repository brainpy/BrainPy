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
import unittest

import brainpy as bp


class TestNormalInit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_normal_init1(self):
        init = bp.init.Normal()
        for size in [(100,), (10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_normal_init2(self):
        init = bp.init.Normal(scale=0.5)
        for size in [(100,), (10, 20)]:
            weights = init(size)
            assert weights.shape == size

    def test_normal_init3(self):
        init1 = bp.init.Normal(scale=0.5, seed=10)
        init2 = bp.init.Normal(scale=0.5, seed=10)
        size = (10,)
        weights1 = init1(size)
        weights2 = init2(size)
        assert weights1.shape == size
        assert (weights1 == weights2).all()


class TestUniformInit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_uniform_init1(self):
        init = bp.init.Normal()
        for size in [(100,), (10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_uniform_init2(self):
        init = bp.init.Uniform(min_val=10, max_val=20)
        for size in [(100,), (10, 20)]:
            weights = init(size)
            assert weights.shape == size


class TestVarianceScaling(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_var_scaling1(self):
        init = bp.init.VarianceScaling(scale=1., mode='fan_in', distribution='truncated_normal')
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_var_scaling2(self):
        init = bp.init.VarianceScaling(scale=2, mode='fan_out', distribution='normal')
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_var_scaling3(self):
        init = bp.init.VarianceScaling(scale=2 / 4, mode='fan_avg', in_axis=0, out_axis=1,
                                       distribution='uniform')
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestKaimingUniformUnit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_kaiming_uniform_init(self):
        init = bp.init.KaimingUniform()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestKaimingNormalUnit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_kaiming_normal_init(self):
        init = bp.init.KaimingNormal()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestXavierUniformUnit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_xavier_uniform_init(self):
        init = bp.init.XavierUniform()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestXavierNormalUnit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_xavier_normal_init(self):
        init = bp.init.XavierNormal()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestLecunUniformUnit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_lecun_uniform_init(self):
        init = bp.init.LecunUniform()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestLecunNormalUnit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_lecun_normal_init(self):
        init = bp.init.LecunNormal()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestOrthogonalUnit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_orthogonal_init1(self):
        init = bp.init.Orthogonal()
        for size in [(20, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_orthogonal_init2(self):
        init = bp.init.Orthogonal(scale=2., axis=0)
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestDeltaOrthogonalUnit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bp.math.random.seed()

    def test_delta_orthogonal_init1(self):
        init = bp.init.DeltaOrthogonal()
        for size in [(20, 20, 20), (10, 20, 30, 40), (50, 40, 30, 20, 20)]:
            weights = init(size)
            assert weights.shape == size
