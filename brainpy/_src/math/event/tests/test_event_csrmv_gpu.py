# -*- coding: utf-8 -*-


import jax
import pytest

import test_event_csrmv

if jax.default_backend() != 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class Test_event_csr_matvec_GPU(test_event_csrmv.Test_event_csr_matvec):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs, platform='gpu')
