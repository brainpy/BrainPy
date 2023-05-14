# -*- coding: utf-8 -*-

import jax
import pytest

import test_event_matvec

if jax.default_backend() != 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class Test_event_matvec_prob_conn_GPU(test_event_matvec.Test_event_matvec_prob_conn):
  def __init__(self, *args, **kwargs):
    super(Test_event_matvec_prob_conn_GPU, self).__init__(*args, **kwargs, platform='gpu')
