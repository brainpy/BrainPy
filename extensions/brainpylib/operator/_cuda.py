import ctypes
import ctypes.util
import sys

from cffi import FFI
from numba import cuda
from numba import types


class Dl_info(ctypes.Structure):
  """
  Structure of the Dl_info returned by the CFFI of dl.dladdr
  """

  _fields_ = (
    ("dli_fname", ctypes.c_char_p),
    ("dli_fbase", ctypes.c_void_p),
    ("dli_sname", ctypes.c_char_p),
    ("dli_saddr", ctypes.c_void_p),
  )


# Find the dynamic linker library path. Only works on unix-like os
libdl_path = ctypes.util.find_library("dl")
if libdl_path:
  # Load the dynamic linker dynamically
  libdl = ctypes.CDLL(libdl_path)

  # Define dladdr to get the pointer to a symbol in a shared
  # library already loaded.
  # https://man7.org/linux/man-pages/man3/dladdr.3.html
  libdl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(Dl_info))
  # restype is None as it returns by reference
else:
  # On Windows it is nontrivial to have libdl, so we disable everything about
  # it and use other ways to find paths of libraries
  libdl = None


def find_path_of_symbol_in_library(symbol):
  if libdl is None:
    raise ValueError("libdl not found.")

  info = Dl_info()
  result = libdl.dladdr(symbol, ctypes.byref(info))
  if result and info.dli_fname:
    return info.dli_fname.decode(sys.getfilesystemencoding())
  else:
    raise ValueError("Cannot determine path of Library.")


try:
  _libcuda = cuda.driver.find_driver()
  if sys.platform == "win32":
    libcuda_path = ctypes.util.find_library(_libcuda._name)
  else:
    libcuda_path = find_path_of_symbol_in_library(_libcuda.cuMemcpy)
  numba_cffi_loaded = True
except Exception:
  numba_cffi_loaded = False


if numba_cffi_loaded:
  # functions needed
  ffi = FFI()
  ffi.cdef("int cuMemcpy(void* dst, void* src, unsigned int len, int type);")
  ffi.cdef("int cuMemcpyAsync(void* dst, void* src, unsigned int len, int type, void* stream);")
  ffi.cdef("int cuStreamSynchronize(void* stream);")
  ffi.cdef("int cudaMallocHost(void** ptr, size_t size);")
  ffi.cdef("int cudaFreeHost(void* ptr);")

  # load libraray
  # could  ncuda.driver.find_library()
  libcuda = ffi.dlopen(libcuda_path)
  cuMemcpy = libcuda.cuMemcpy
  cuMemcpyAsync = libcuda.cuMemcpyAsync
  cuStreamSynchronize = libcuda.cuStreamSynchronize

  memcpyHostToHost = types.int32(0)
  memcpyHostToDevice = types.int32(1)
  memcpyDeviceToHost = types.int32(2)
  memcpyDeviceToDevice = types.int32(3)
