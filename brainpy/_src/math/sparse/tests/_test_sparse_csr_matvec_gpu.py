# -*- coding: utf-8 -*-

import jax
import pytest
import test_sparse_csr_matvec

if jax.default_backend() != 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class Test_csr_matvec_GPU(test_sparse_csr_matvec.Test_csr_matvec):
  def __init__(self, *args, **kwargs):
    super(Test_csr_matvec_GPU, self).__init__(*args, **kwargs, platform='gpu')

