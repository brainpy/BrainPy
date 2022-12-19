# -*- coding: utf-8 -*-


from brainpy.errors import PackageMissingError

__all__ = [
  'is_checking',
  'turn_on',
  'turn_off',
]

_check = True

_BRAINPYLIB_MINIMAL_VERSION = '0.1.3'


try:
  import jaxlib
  del jaxlib
except ModuleNotFoundError:
  raise ModuleNotFoundError(
    '''

BrainPy needs jaxlib, please install it. 

1. If you are using Windows system, install jaxlib through

   >>> pip install jaxlib -f https://whls.blob.core.windows.net/unstable/index.html

2. If you are using macOS platform, install jaxlib through

   >>> pip install jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html

3. If you are using Linux platform, install jaxlib through

   >>> pip install jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html

4. If you are using Linux + CUDA platform, install jaxlib through

   >>> pip install jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Note that the versions of "jax" and "jaxlib" should be consistent, like "jax=0.3.14" and "jaxlib=0.3.14".  

For more detail installation instructions, please see https://brainpy.readthedocs.io/en/latest/quickstart/installation.html#dependency-2-jax 

    ''') from None


try:
  import brainpylib

  if brainpylib.__version__ < _BRAINPYLIB_MINIMAL_VERSION:
    raise PackageMissingError(
      f'brainpy need "brainpylib>={_BRAINPYLIB_MINIMAL_VERSION}". \n'
      f'Please install it through:\n\n'
      f'>>> pip install brainpylib -U'
    )

  del brainpylib
except ModuleNotFoundError:
  raise PackageMissingError(
    f'brainpy need "brainpylib>={_BRAINPYLIB_MINIMAL_VERSION}". \n'
    f'Please install "brainpylib>={_BRAINPYLIB_MINIMAL_VERSION}" through:\n\n'
    f'>>> pip install brainpylib'
  )


def is_checking():
  """Whether the checking is turn on."""
  return _check


def turn_on():
  """Turn on the checking."""
  global _check
  _check = True


def turn_off():
  """Turn off the checking."""
  global _check
  _check = False



