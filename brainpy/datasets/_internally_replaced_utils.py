# -*- coding: utf-8 -*-

import ctypes
import errno
import hashlib
import importlib.machinery
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from brainpy import math as bm

from tqdm import tqdm

ENV_TORCH_HOME = 'BRAINPY_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


def _get_torch_home():
  torch_home = os.path.expanduser(
    os.getenv(ENV_TORCH_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'brainpy')))
  return torch_home


# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
_HOME = os.path.join(_get_torch_home(), "datasets", "vision")
_USE_SHARDED_DATASETS = False


def _download_file_from_remote_location(fpath: str, url: str) -> None:
  pass


def _is_remote_location_available() -> bool:
  return False


def get_dir():
  r"""
  Get the Torch Hub cache directory used for storing downloaded models & weights.

  If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
  environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
  ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
  filesystem layout, with a default value ``~/.cache`` if the environment
  variable is not set.
  """
  # Issue warning to move data if old env is set
  return os.path.join(_get_torch_home(), 'hub')


def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
  r"""Loads the Torch serialized object at the given URL.

  If downloaded file is a zip file, it will be automatically
  decompressed.

  If the object is already present in `model_dir`, it's deserialized and
  returned.
  The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
  ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

  Args:
      url (string): URL of the object to download
      model_dir (string, optional): directory in which to save the object
      map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
      progress (bool, optional): whether or not to display a progress bar to stderr.
          Default: True
      check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
          ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
          digits of the SHA256 hash of the contents of the file. The hash is used to
          ensure unique names and to verify the contents of the file.
          Default: False
      file_name (string, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

  Example:
      >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

  """
  # Issue warning to move data if old env is set
  if os.getenv('TORCH_MODEL_ZOO'):
    warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

  if model_dir is None:
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')

  try:
    os.makedirs(model_dir)
  except OSError as e:
    if e.errno == errno.EEXIST:
      # Directory already exists, ignore.
      pass
    else:
      # Unexpected OSError, re-raise.
      raise

  parts = urlparse(url)
  filename = os.path.basename(parts.path)
  if file_name is not None:
    filename = file_name
  cached_file = os.path.join(model_dir, filename)
  if not os.path.exists(cached_file):
    sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
    hash_prefix = None
    if check_hash:
      r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
      hash_prefix = r.group(1) if r else None
    download_url_to_file(url, cached_file, hash_prefix, progress=progress)

  if _is_legacy_zip_format(cached_file):
    return _legacy_zip_load(cached_file, model_dir, map_location)
  return bm.load(cached_file, map_location=map_location)


def _legacy_zip_load(filename, model_dir, map_location):
  warnings.warn('Falling back to the old format < 1.6. This support will be '
                'deprecated in favor of default zipfile format introduced in 1.6. '
                'Please redo torch.save() to save it in the new zipfile format.')
  # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
  #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
  #       E.g. resnet18-5c106cde.pth which is widely used.
  with zipfile.ZipFile(filename) as f:
    members = f.infolist()
    if len(members) != 1:
      raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
    f.extractall(model_dir)
    extraced_name = members[0].filename
    extracted_file = os.path.join(model_dir, extraced_name)
  return bm.load(extracted_file, map_location=map_location)


# Hub used to support automatically extracts from zipfile manually compressed by users.
# The legacy zip format expects only one file from torch.save() < 1.6 in the zip.
# We should remove this support since zipfile is now default zipfile format for torch.save().
def _is_legacy_zip_format(filename):
  if zipfile.is_zipfile(filename):
    infolist = zipfile.ZipFile(filename).infolist()
    return len(infolist) == 1 and not infolist[0].is_dir()
  return False


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
  r"""Download object at the given URL to a local path.

  Args:
      url (string): URL of the object to download
      dst (string): Full path where object will be saved, e.g. ``/tmp/temporary_file``
      hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
          Default: None
      progress (bool, optional): whether or not to display a progress bar to stderr
          Default: True

  Example:
      >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

  """
  file_size = None
  req = Request(url, headers={"User-Agent": "torch.hub"})
  u = urlopen(req)
  meta = u.info()
  if hasattr(meta, 'getheaders'):
    content_length = meta.getheaders("Content-Length")
  else:
    content_length = meta.get_all("Content-Length")
  if content_length is not None and len(content_length) > 0:
    file_size = int(content_length[0])

  # We deliberately save it in a temp file and move it after
  # download is complete. This prevents a local working checkpoint
  # being overridden by a broken download.
  dst = os.path.expanduser(dst)
  dst_dir = os.path.dirname(dst)
  f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

  try:
    if hash_prefix is not None:
      sha256 = hashlib.sha256()
    with tqdm(total=file_size, disable=not progress,
              unit='B', unit_scale=True, unit_divisor=1024) as pbar:
      while True:
        buffer = u.read(8192)
        if len(buffer) == 0:
          break
        f.write(buffer)
        if hash_prefix is not None:
          sha256.update(buffer)
        pbar.update(len(buffer))

    f.close()
    if hash_prefix is not None:
      digest = sha256.hexdigest()
      if digest[:len(hash_prefix)] != hash_prefix:
        raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                           .format(hash_prefix, digest))
    shutil.move(f.name, dst)
  finally:
    f.close()
    if os.path.exists(f.name):
      os.remove(f.name)


def _get_extension_path(lib_name):
  lib_dir = os.path.dirname(__file__)
  if os.name == "nt":
    # Register the main torchvision library location on the default DLL path
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
    prev_error_mode = kernel32.SetErrorMode(0x0001)

    if with_load_library_flags:
      kernel32.AddDllDirectory.restype = ctypes.c_void_p

    if sys.version_info >= (3, 8):
      os.add_dll_directory(lib_dir)
    elif with_load_library_flags:
      res = kernel32.AddDllDirectory(lib_dir)
      if res is None:
        err = ctypes.WinError(ctypes.get_last_error())
        err.strerror += f' Error adding "{lib_dir}" to the DLL directories.'
        raise err

    kernel32.SetErrorMode(prev_error_mode)

  loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

  extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
  ext_specs = extfinder.find_spec(lib_name)
  if ext_specs is None:
    raise ImportError

  return ext_specs.origin
