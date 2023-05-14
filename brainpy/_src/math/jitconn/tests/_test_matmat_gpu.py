# -*- coding: utf-8 -*-

import jax
import pytest

import matmat_testcase

if jax.default_backend() != 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class Test_matmat_prob_conn_GPU(matmat_testcase.Test_matmat_prob_conn):
  def __init__(self, *args, **kwargs):
    super(Test_matmat_prob_conn_GPU, self).__init__(*args, **kwargs, platform='gpu')

