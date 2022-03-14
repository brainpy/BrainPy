# -*- coding: utf-8 -*-

import jax.numpy.fft

from brainpy.math.jaxarray import JaxArray
from brainpy.math.numpy_ops import _remove_jaxarray

__all__ = [
  "fft", "fft2", "fftfreq", "fftn", "fftshift", "hfft",
  "ifft", "ifft2", "ifftn", "ifftshift", "ihfft", "irfft",
  "irfft2", "irfftn", "rfft", "rfft2", "rfftfreq", "rfftn"
]


def fft(a, n=None, axis=-1, norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.fft(a=a, n=n, axis=axis, norm=norm))


def fft2(a, s=None, axes=(-2, -1), norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.fft2(a=a, s=s, axes=axes, norm=norm))


def fftfreq(n, d=1.0):
  return JaxArray(jax.numpy.fft.fftfreq(n=n, d=d))


def fftn(a, s=None, axes=None, norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.fftn(a=a, s=s, axes=axes, norm=norm))


def fftshift(x, axes=None):
  x = _remove_jaxarray(x)
  return JaxArray(jax.numpy.fft.fftshift(x=x, axes=axes))


def hfft(a, n=None, axis=-1, norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.hfft(a=a, n=n, axis=axis, norm=norm))


def ifft(a, n=None, axis=-1, norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.ifft(a=a, n=n, axis=axis, norm=norm))


def ifft2(a, s=None, axes=(-2, -1), norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.ifft2(a=a, s=s, axes=axes, norm=norm))


def ifftn(a, s=None, axes=None, norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.ifftn(a=a, s=s, axes=axes, norm=norm))


def ifftshift(x, axes=None):
  x = _remove_jaxarray(x)
  return JaxArray(jax.numpy.fft.ifftshift(x=x, axes=axes))


def ihfft(a, n=None, axis=-1, norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.ihfft(a=a, n=n, axis=axis, norm=norm))


def irfft(a, n=None, axis=-1, norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.irfft(a=a, n=n, axis=axis, norm=norm))


def irfft2(a, s=None, axes=(-2, -1), norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.irfft2(a=a, s=s, axes=axes, norm=norm))


def irfftn(a, s=None, axes=None, norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.irfftn(a=a, s=s, axes=axes, norm=norm))


def rfft(a, n=None, axis=-1, norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.rfft(a=a, n=n, axis=axis, norm=norm))


def rfft2(a, s=None, axes=(-2, -1), norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.rfft2(a=a, s=s, axes=axes, norm=norm))


def rfftfreq(n, d=1.0):
  return JaxArray(jax.numpy.fft.rfftfreq(n=n, d=d))


def rfftn(a, s=None, axes=None, norm=None):
  a = _remove_jaxarray(a)
  return JaxArray(jax.numpy.fft.rfftn(a=a, s=s, axes=axes, norm=norm))
