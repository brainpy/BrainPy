# -*- coding: utf-8 -*-

import jax
import pytest

import test_info

if jax.default_backend() != 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class Test_event_info_GPU(test_info.Test_event_info):
  def __init__(self, *args, **kwargs):
    super(Test_event_info_GPU, self).__init__(*args, **kwargs, platform='gpu')
