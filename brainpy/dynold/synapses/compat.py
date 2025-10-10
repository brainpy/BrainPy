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
import warnings
from typing import Union, Dict, Callable

from brainpy.connect import TwoEndConnector
from brainpy.dyn.base import NeuDyn
from brainpy.dynold.synouts import COBA, CUBA
from brainpy.initialize import Initializer
from brainpy.types import ArrayType
from .abstract_models import Delta, Exponential, DualExponential

__all__ = [
    'DeltaSynapse',
    'ExpCUBA',
    'ExpCOBA',
    'DualExpCUBA',
    'DualExpCOBA',
    'AlphaCUBA',
    'AlphaCOBA',
]


class DeltaSynapse(Delta):
    """Delta synapse.

    .. deprecated:: 2.1.13
       Please use "brainpy.synapses.Delta" instead.

    """

    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
        conn_type: str = 'sparse',
        weights: Union[float, ArrayType, Initializer, Callable] = 1.,
        delay_step: Union[float, ArrayType, Initializer, Callable] = None,
        post_input_key: str = 'V',
        post_has_ref: bool = False,
        name: str = None,
    ):
        warnings.warn('Please use "brainpy.synapses.Delta" instead.', DeprecationWarning)
        super().__init__(pre=pre,
                         post=post,
                         conn=conn,
                         output=CUBA(post_input_key),
                         name=name,
                         comp_method=conn_type,
                         g_max=weights,
                         delay_step=delay_step,
                         post_ref_key='refractory' if post_has_ref else None)


class ExpCUBA(Exponential):
    r"""Current-based exponential decay synapse model.

    .. deprecated:: 2.1.13
       Please use "brainpy.synapses.Exponential" instead.

    """

    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
        conn_type: str = 'sparse',
        g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
        delay_step: Union[int, ArrayType, Initializer, Callable] = None,
        tau: Union[float, ArrayType] = 8.0,
        name: str = None,
        method: str = 'exp_auto',
    ):
        super().__init__(pre=pre,
                         post=post,
                         conn=conn,
                         name=name,
                         comp_method=conn_type,
                         g_max=g_max,
                         delay_step=delay_step,
                         tau=tau,
                         method=method,
                         output=CUBA())


class ExpCOBA(Exponential):
    """Conductance-based exponential decay synapse model.

    .. deprecated:: 2.1.13
       Please use "brainpy.synapses.Exponential" instead.
    """

    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        # connection
        conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
        conn_type: str = 'sparse',
        # connection strength
        g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
        # synapse parameter
        tau: Union[float, ArrayType] = 8.0,
        E: Union[float, ArrayType] = 0.,
        # synapse delay
        delay_step: Union[int, ArrayType, Initializer, Callable] = None,
        # others
        method: str = 'exp_auto',
        name: str = None
    ):
        super().__init__(pre=pre,
                         post=post,
                         conn=conn,
                         comp_method=conn_type,
                         g_max=g_max,
                         delay_step=delay_step,
                         tau=tau,
                         method=method,
                         name=name,
                         output=COBA(E=E))


class DualExpCUBA(DualExponential):
    r"""Current-based dual exponential synapse model.

    .. deprecated:: 2.1.13
       Please use "brainpy.synapses.DualExponential" instead.

    """

    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
        conn_type: str = 'dense',
        g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
        tau_decay: Union[float, ArrayType] = 10.0,
        tau_rise: Union[float, ArrayType] = 1.,
        delay_step: Union[int, ArrayType, Initializer, Callable] = None,
        method: str = 'exp_auto',
        name: str = None
    ):
        super().__init__(pre=pre,
                         post=post,
                         conn=conn,
                         comp_method=conn_type,
                         g_max=g_max,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         delay_step=delay_step,
                         method=method,
                         name=name,
                         output=CUBA())


class DualExpCOBA(DualExponential):
    """Conductance-based dual exponential synapse model.


    .. deprecated:: 2.1.13
       Please use "brainpy.synapses.DualExponential" instead.

    """

    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
        conn_type: str = 'dense',
        g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
        delay_step: Union[int, ArrayType, Initializer, Callable] = None,
        tau_decay: Union[float, ArrayType] = 10.0,
        tau_rise: Union[float, ArrayType] = 1.,
        E: Union[float, ArrayType] = 0.,
        method: str = 'exp_auto',
        name: str = None
    ):
        super().__init__(pre=pre,
                         post=post,
                         conn=conn,
                         comp_method=conn_type,
                         g_max=g_max,
                         tau_decay=tau_decay,
                         tau_rise=tau_rise,
                         delay_step=delay_step,
                         method=method,
                         name=name,
                         output=COBA(E=E))


class AlphaCUBA(DualExpCUBA):
    r"""Current-based alpha synapse model.

    .. deprecated:: 2.1.13
       Please use "brainpy.synapses.Alpha" instead.

    """

    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
        conn_type: str = 'dense',
        g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
        delay_step: Union[int, ArrayType, Initializer, Callable] = None,
        tau_decay: Union[float, ArrayType] = 10.0,
        method: str = 'exp_auto',
        name: str = None
    ):
        super().__init__(pre=pre,
                         post=post,
                         conn=conn,
                         conn_type=conn_type,
                         delay_step=delay_step,
                         g_max=g_max,
                         tau_decay=tau_decay,
                         tau_rise=tau_decay,
                         method=method,
                         name=name)


class AlphaCOBA(DualExpCOBA):
    """Conductance-based alpha synapse model.

    .. deprecated:: 2.1.13
       Please use "brainpy.synapses.Alpha" instead.

    """

    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
        conn_type: str = 'dense',
        g_max: Union[float, ArrayType, Callable, Initializer] = 1.,
        delay_step: Union[int, ArrayType, Initializer, Callable] = None,
        tau_decay: Union[float, ArrayType] = 10.0,
        E: Union[float, ArrayType] = 0.,
        method: str = 'exp_auto',
        name: str = None
    ):
        super().__init__(pre=pre,
                         post=post,
                         conn=conn,
                         conn_type=conn_type,
                         delay_step=delay_step,
                         g_max=g_max, E=E,
                         tau_decay=tau_decay,
                         tau_rise=tau_decay,
                         method=method,
                         name=name)
