# -*- coding: utf-8 -*-

from brainpy.errors import PackageMissingError

try:
  import brainpylib
except ModuleNotFoundError:
  brainpylib = None


_BRAINPYLIB_MINIMAL_VERSION = '0.1.0'


def _check_brainpylib(ops_name):
  if brainpylib is not None:
    if brainpylib.__version__ < _BRAINPYLIB_MINIMAL_VERSION:
      raise PackageMissingError(
        f'"{ops_name}" operator need "brainpylib>={_BRAINPYLIB_MINIMAL_VERSION}". \n'
        f'Please install it through:\n\n'
        f'>>> pip install brainpylib>={_BRAINPYLIB_MINIMAL_VERSION}\n'
        f'>>> # or \n'
        f'>>> pip install brainpylib -U'
      )
  else:
    raise PackageMissingError(
      f'"brainpylib" must be installed when the user '
      f'wants to use "{ops_name}" operator. \n'
      f'Please install "brainpylib>={_BRAINPYLIB_MINIMAL_VERSION}" through:\n\n'
      f'>>> pip install brainpylib'
    )
