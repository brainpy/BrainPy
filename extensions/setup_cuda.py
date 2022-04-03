import distutils.sysconfig as sysconfig
import os
import platform
import re
import subprocess
import sys
import glob
import pybind11
from setuptools import find_packages, setup, Extension
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
      # "-DPYTHON_LIBRARY={}".format(os.path.join(sysconfig.get_config_var('LIBDIR'),
      #                                           sysconfig.get_config_var('LDLIBRARY'))),
      # "-DPYTHON_INCLUDE_DIRS={}".format(sysconfig.get_python_inc()),
      #"-DPYTHON_INCLUDE_DIR={}".format(sysconfig.get_python_inc()),
      "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
      #"-DPython_EXECUTABLE={}".format(sys.executable),
      #"-DPython_LIBRARIES={}".format(cmake_python_library),
      #"-DPython_INCLUDE_DIRS={}".format(cmake_python_include_dir),
      # "-DCMAKE_BUILD_TYPE={}".format("Debug" if self.debug else "Release"),
      # "-DCMAKE_PREFIX_PATH={}".format(pybind11.get_cmake_dir()),
      # "-DCMAKE_CUDA_FLAGS={}".format('"-arch=sm_61"')
    ]
    if os.environ.get("BRAINPY_CUDA", "no").lower() == "yes":
      cmake_args.append("-BRAINPY_CUDA=yes")
    print(" ".join(cmake_args))

    os.makedirs(self.build_temp, exist_ok=True)
    subprocess.check_call(
      ["cmake", '-DCMAKE_CUDA_FLAGS="-arch=sm_61"', HERE] + cmake_args, cwd=self.build_temp
    )

    # Build all the extensions
    super().build_extensions()

    # Finally run install
    subprocess.check_call(["cmake", "--build", ".", "--target", "install"],
                          cwd=self.build_temp)

  def build_extension(self, ext):
    # target_name = ext.name.split(".")[-1]
    #subprocess.check_call(
    #  ["cmake", "."], cwd=self.build_temp)
    subprocess.check_call(["cmake", "--build", ".", "--target", "gpu_ops"],
                          cwd=self.build_temp)

# version control
with open(os.path.join(HERE, 'brainpylib', '__init__.py'), 'r') as f:
  init_py = f.read()
  __version__ = re.search('__version__ = "(.*)"', init_py).groups()[0]

cuda_version = os.environ.get("JAX_CUDA_VERSION")
if cuda_version:
  __version__ += "+cuda" + cuda_version.replace(".", "")

# build
setup(
  name='brainpylib',
  version=__version__,
  description='C++/CUDA Library for BrainPy',
  author='BrainPy team',
  author_email='chao.brain@qq.com',
  packages=find_packages(exclude=['lib*']),
  include_package_data=True,
  install_requires=["jax", "jaxlib", "pybind11>=2.6, <2.8", "cffi", "numba"],
  extras_require={"test": "pytest"},
  python_requires='>=3.7',
  url='https://github.com/PKU-NIP-Lab/BrainPy',
  ext_modules=[
    Extension("gpu_ops", ['lib/gpu_ops.cc'] + glob.glob("lib/*.cu")),
    Extension("cpu_ops", ['lib/cpu_ops.cc'] + glob.glob("lib/*.cc")),
  ],
  cmdclass={"build_ext": CMakeBuildExt},
  license='GPL-3.0 License',
)