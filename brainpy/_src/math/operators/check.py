# -*- coding: utf-8 -*-


jaxlib_minimal_version = '0.3.14'

jax_install_msg = '''

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

'''

try:
  import jaxlib
except (ModuleNotFoundError, ImportError):
  raise ModuleNotFoundError(f'brainpylib needs jaxlib >= {jaxlib_minimal_version}, please install it. '
                            + jax_install_msg)

if jaxlib.__version__ < jaxlib_minimal_version:
  raise RuntimeError(f'brainpylib needs jaxlib >= {jaxlib_minimal_version}, please upgrade it. '
                     + jax_install_msg)
