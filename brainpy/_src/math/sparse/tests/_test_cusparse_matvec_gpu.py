# -*- coding: utf-8 -*-

import jax
import pytest

import test_cusparse_matvec

if jax.default_backend() != 'gpu':
  pytest.skip("No gpu available.", allow_module_level=True)


class Test_cusparse_csr_matvec_GPU(test_cusparse_matvec.Test_cusparse_csr_matvec):
  def __init__(self, *args, **kwargs):
    super(Test_cusparse_csr_matvec_GPU, self).__init__(*args, **kwargs, platform='gpu')


