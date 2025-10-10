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

import pytest

from brainpy import _errors as errors
from brainpy import odeint


class TestAdaptiveRKKeywords(unittest.TestCase):
    def test_bs(self):
        method = 'bs'
        for adaptive in [True, False]:

            print(f'Test {"adaptive" if adaptive else ""} {method} method:')
            print()
            odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_t_arg: t)

            if adaptive:
                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

            print('-' * 40)

    def test_rkf45(self):
        method = 'rkf45'
        for adaptive in [True, False]:

            print(f'Test {"adaptive" if adaptive else ""} {method} method:')
            print()
            odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_t_arg: t)

            if adaptive:
                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

            print('-' * 40)

    def test_heun_euler(self):
        method = 'heun_euler'
        for adaptive in [True, False]:

            print(f'Test {"adaptive" if adaptive else ""} {method} method:')
            print()
            odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

            if adaptive:
                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

            print('-' * 40)

    def test_rkf12(self):
        method = 'rkf12'
        for adaptive in [True, False]:

            print(f'Test {"adaptive" if adaptive else ""} {method} method:')
            print()
            odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_t_arg: t)

            if adaptive:
                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

            print('-' * 40)

    def test_ck(self):
        method = 'ck'
        for adaptive in [True, False]:

            print(f'Test {"adaptive" if adaptive else ""} {method} method:')
            print()
            odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k5_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k5_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k6_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k6_t_arg: t)

            if adaptive:
                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

            print('-' * 40)

    def test_rkdp(self):
        method = 'rkdp'
        for adaptive in [True, False]:

            print(f'Test {"adaptive" if adaptive else ""} {method} method:')
            print()
            odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, p: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda f, t, dt: t)

            # with pytest.raises(errors.CodeError):
            #   odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_new: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k1: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dv_k2: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k2_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k3_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k4_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k5_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k5_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k6_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k6_t_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k7_v_arg: t)

            with pytest.raises(errors.CodeError):
                odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, k7_t_arg: t)

            if adaptive:
                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, error: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, v_te: t)

                with pytest.raises(errors.CodeError):
                    odeint(method=method, show_code=True, adaptive=adaptive, f=lambda v, t, dt_new: t)

            print('-' * 40)
