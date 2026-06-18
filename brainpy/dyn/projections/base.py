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
# Re-export the real projection base classes. This module previously held a
# byte-for-byte duplicate of the private ``_get_return`` helper in ``utils.py``
# (H-40); that helper is private, so ``from .base import *`` exported nothing and
# the duplicate was both dead and misleading versus the real base classes.
from brainpy.dynsys import Projection
from .conn import SynConn

__all__ = ['Projection', 'SynConn']
