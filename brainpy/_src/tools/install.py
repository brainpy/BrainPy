
__all__ = [
  'jaxlib_install_info',
]


jaxlib_install_info = '''

BrainPy needs jaxlib, please install it. 

1. If you are using brainpy on CPU platform, please install jaxlib through

   >>> pip install jaxlib 

2. If you are using Linux + CUDA platform, install jaxlib through

   >>> pip install jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Note that the versions of "jax" and "jaxlib" should be consistent, like "jax=0.3.14" and "jaxlib=0.3.14".  

For more detail installation instructions, please see https://brainpy.readthedocs.io/en/latest/quickstart/installation.html#dependency-2-jax 

'''
