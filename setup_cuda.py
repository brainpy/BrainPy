# -*- coding: utf-8 -*-

import distutils.sysconfig as sysconfig
import glob
import os
import platform
import re
import subprocess
import sys
import io

try:
  import pybind11
except ModuleNotFoundError:
  raise ModuleNotFoundError('Please install pybind11 before installing brainpy!')
from setuptools import find_packages, setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.realpath(__file__))


# This custom class for building the extensions uses CMake to compile. You
# don't have to use CMake for this task, but I found it to be the easiest when
# compiling ops with GPU support since setuptools doesn't have great CUDA
# support.
class CMakeBuildExt(build_ext):
  def build_extensions(self):
    # Work out the relevant Python paths to pass to CMake,
    # adapted from the PyTorch build system
    if platform.system() == "Windows":
      cmake_python_library = "{}/libs/python{}.lib".format(
        sysconfig.get_config_var("prefix"),
        sysconfig.get_config_var("VERSION"),
      )
      if not os.path.exists(cmake_python_library):
        cmake_python_library = "{}/libs/python{}.lib".format(
          sys.base_prefix,
          sysconfig.get_config_var("VERSION"),
        )
    else:
      cmake_python_library = "{}/{}".format(sysconfig.get_config_var("LIBDIR"),
                                            sysconfig.get_config_var("INSTSONAME"))
    cmake_python_include_dir = sysconfig.get_python_inc()
    install_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath("dummy")))
    print("install_dir", install_dir)
    os.makedirs(install_dir, exist_ok=True)
    cmake_args = [
      "-DPYTHON_LIBRARY={}".format(os.path.join(sysconfig.get_config_var('LIBDIR'))),
      "-DPYTHON_INCLUDE_DIRS={}".format(sysconfig.get_python_inc()),
      "-DPYTHON_INCLUDE_DIR={}".format(sysconfig.get_python_inc()),
      "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
      "-DPython_EXECUTABLE={}".format(sys.executable),
      "-DPython_LIBRARIES={}".format(cmake_python_library),
      "-DPython_INCLUDE_DIRS={}".format(cmake_python_include_dir),
      "-DCMAKE_BUILD_TYPE={}".format("Debug" if self.debug else "Release"),
      "-DCMAKE_PREFIX_PATH={}".format(os.path.dirname(pybind11.get_cmake_dir())),
      "-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda",
      "-DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so",
      "-DCMAKE_CUDA_FLAGS={}".format('-arch=sm_52 '
                                     '-gencode=arch=compute_52,code=sm_52 '
                                     '-gencode=arch=compute_60,code=sm_60 '
                                     '-gencode=arch=compute_61,code=sm_61 '
                                     '-gencode=arch=compute_70,code=sm_70 '
                                     '-gencode=arch=compute_75,code=sm_75 '
                                     '-gencode=arch=compute_80,code=sm_80 '
                                     '-gencode=arch=compute_86,code=sm_86 '
                                     '-gencode=arch=compute_87,code=sm_87 '
                                     '-gencode=arch=compute_86,code=compute_86'),
      "-DCUDACXX=/usr/local/cuda/bin/nvcc",
      # "-DCMAKE_CUDA_ARCHITECTURES={}".format(86)
    ]
    if os.environ.get("BRAINPY_CUDA", "no").lower() == "yes":
      cmake_args.append("-BRAINPY_CUDA=yes")
    print(" ".join(cmake_args))

    os.makedirs(self.build_temp, exist_ok=True)
    # subprocess.check_call(["cmake", '-DCMAKE_CUDA_FLAGS="-arch=sm_86"'] + cmake_args + [HERE],
    #                       cwd=self.build_temp)
    subprocess.check_call(["cmake"] + cmake_args + [HERE],
                          cwd=self.build_temp)

    # Build all the extensions
    super().build_extensions()

    # Finally run install
    subprocess.check_call(["cmake", "--build", ".", "--target", "install"], cwd=self.build_temp)

  def build_extension(self, ext):
    subprocess.check_call(["cmake", "--build", ".", "--target", "gpu_ops"], cwd=self.build_temp)

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
  README = f.read()


# version control
with open(os.path.join(HERE, 'brainpy', '__init__.py'), 'r') as f:
  init_py = f.read()
  __version__ = re.search('__version__ = "(.*)"', init_py).groups()[0]

# cuda_version = "11_6"
# # if cuda_version:
# __version__ += "+cuda" + cuda_version
# print(__version__)

# build
setup(
  name='brainpy',
  version=__version__,
  description='BrainPy: Brain Dynamics Programming in Python',
  long_description=README,
  long_description_content_type="text/markdown",
  author='BrainPy team',
  author_email='chao.brain@qq.com',
  packages=find_namespace_packages(exclude=['lib*', 'docs*', 'tests*', 'examples*']),
  include_package_data=True,
  install_requires=['numpy>=1.15', 'jax>=0.3.0', 'tqdm', 'msgpack', "numba", "numpy"],
  extras_require={"test": "pytest"},
  python_requires='>=3.7',
  url='https://github.com/brainpy/brainpy',
  ext_modules=[
    Extension("gpu_ops", ['lib/gpu_ops.cc'] + glob.glob("lib/*.cu")),
    Extension("cpu_ops", glob.glob("lib/cpu_*.cc") + glob.glob("lib/cpu_*.cpp")),
  ],
  cmdclass={"build_ext": CMakeBuildExt},
  license='GPL-3.0 license',
  keywords=('event-driven computation, '
            'sparse computation, '
            'brainpy'),
  classifiers=[
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries',
  ],
)
