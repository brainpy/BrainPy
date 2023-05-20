# -*- coding: utf-8 -*-

import jax
import pytest

import test_csrmv

if jax.default_backend() != 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class Test_cusparse_csrmv_GPU(test_csrmv.Test_cusparse_csrmv):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs, platform='gpu')


class Test__csrmv_GPU(test_csrmv.Test_csrmv):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs, platform='gpu')


