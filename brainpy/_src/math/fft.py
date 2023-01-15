# -*- coding: utf-8 -*-

import jax.numpy.fft as jfft

from ._utils import _compatible_with_brainpy_array

__all__ = [
  "fft", "fft2", "fftfreq", "fftn", "fftshift", "hfft",
  "ifft", "ifft2", "ifftn", "ifftshift", "ihfft", "irfft",
  "irfft2", "irfftn", "rfft", "rfft2", "rfftfreq", "rfftn"
]

fft = _compatible_with_brainpy_array(jfft.fft)
fft2 = _compatible_with_brainpy_array(jfft.fft2)
fftfreq = _compatible_with_brainpy_array(jfft.fftfreq)
fftn = _compatible_with_brainpy_array(jfft.fftn)
fftshift = _compatible_with_brainpy_array(jfft.fftshift)
hfft = _compatible_with_brainpy_array(jfft.hfft)
ifft = _compatible_with_brainpy_array(jfft.ifft)
ifft2 = _compatible_with_brainpy_array(jfft.ifft2)
ifftn = _compatible_with_brainpy_array(jfft.ifftn)
ifftshift = _compatible_with_brainpy_array(jfft.ifftshift)
ihfft = _compatible_with_brainpy_array(jfft.ihfft)
irfft = _compatible_with_brainpy_array(jfft.irfft)
irfft2 = _compatible_with_brainpy_array(jfft.irfft2)
irfftn = _compatible_with_brainpy_array(jfft.irfftn)
rfft = _compatible_with_brainpy_array(jfft.rfft)
rfft2 = _compatible_with_brainpy_array(jfft.rfft2)
rfftfreq = _compatible_with_brainpy_array(jfft.rfftfreq)
rfftn = _compatible_with_brainpy_array(jfft.rfftn)
