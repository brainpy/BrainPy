# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import jax.numpy.fft as jfft

from ._utils import _compatible_with_brainpy_array

__all__ = [
    "fft", "fft2", "fftfreq", "fftn", "fftshift", "hfft",
    "ifft", "ifft2", "ifftn", "ifftshift", "ihfft", "irfft",
    "irfft2", "irfftn", "rfft", "rfft2", "rfftfreq", "rfftn"
]

fft = _compatible_with_brainpy_array(jfft.fft, module='fft.')
fft2 = _compatible_with_brainpy_array(jfft.fft2, module='fft.')
fftfreq = _compatible_with_brainpy_array(jfft.fftfreq, module='fft.')
fftn = _compatible_with_brainpy_array(jfft.fftn, module='fft.')
fftshift = _compatible_with_brainpy_array(jfft.fftshift, module='fft.')
hfft = _compatible_with_brainpy_array(jfft.hfft, module='fft.')
ifft = _compatible_with_brainpy_array(jfft.ifft, module='fft.')
ifft2 = _compatible_with_brainpy_array(jfft.ifft2, module='fft.')
ifftn = _compatible_with_brainpy_array(jfft.ifftn, module='fft.')
ifftshift = _compatible_with_brainpy_array(jfft.ifftshift, module='fft.')
ihfft = _compatible_with_brainpy_array(jfft.ihfft, module='fft.')
irfft = _compatible_with_brainpy_array(jfft.irfft, module='fft.')
irfft2 = _compatible_with_brainpy_array(jfft.irfft2, module='fft.')
irfftn = _compatible_with_brainpy_array(jfft.irfftn, module='fft.')
rfft = _compatible_with_brainpy_array(jfft.rfft, module='fft.')
rfft2 = _compatible_with_brainpy_array(jfft.rfft2, module='fft.')
rfftfreq = _compatible_with_brainpy_array(jfft.rfftfreq, module='fft.')
rfftn = _compatible_with_brainpy_array(jfft.rfftn, module='fft.')
