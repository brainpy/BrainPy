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
import numbers
from typing import Dict, Optional, Union, Callable

import jax
import jax.numpy as jnp
import numpy as np
from brainevent import csr_on_pre, csr2csc_on_post
from brainevent import dense_on_pre, dense_on_post

from brainpy import connect, initialize as init
from brainpy import math as bm
from brainpy._errors import MathError
from brainpy.check import is_initializer
from brainpy.connect import csr2csc
from brainpy.context import share
from brainpy.dnn.base import Layer
from brainpy.initialize import XavierNormal, ZeroInit, Initializer, parameter
from brainpy.mixin import SupportOnline, SupportOffline, SupportSTDP
from brainpy.types import ArrayType, Sharding

__all__ = [
    'Dense', 'Linear',
    'Identity',
    'AllToAll',
    'OneToOne',
    'MaskedLinear',
    'CSRLinear', 'EventCSRLinear',
    'JitFPHomoLinear', 'JitFPUniformLinear', 'JitFPNormalLinear',
    'EventJitFPHomoLinear', 'EventJitFPNormalLinear', 'EventJitFPUniformLinear',
]


class Dense(Layer, SupportSTDP, SupportOnline, SupportOffline):
    r"""A linear transformation applied over the last dimension of the input.

    Mathematically, this node can be defined as:

    .. math::

       y = x  \cdot weight + b

    Parameters::

    num_in: int
      The number of the input feature. A positive integer.
    num_out: int
      The number of the output features. A positive integer.
    W_initializer: optional, Initializer
      The weight initialization.
    b_initializer: optional, Initializer
      The bias initialization.
    mode: Mode
      Enable training this node or not. (default True)
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        W_initializer: Union[Initializer, Callable, ArrayType] = XavierNormal(),
        b_initializer: Optional[Union[Initializer, Callable, ArrayType]] = ZeroInit(),
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super(Dense, self).__init__(mode=mode, name=name)

        # shape
        self.num_in = num_in
        self.num_out = num_out
        if num_in < 0:
            raise ValueError(f'Received an invalid value for `num_out`, expected '
                             f'a positive integer. Received: num_in={num_in}')
        if num_out < 0:
            raise ValueError(f'Received an invalid value for `num_out`, expected '
                             f'a positive integer. Received: num_out={num_out}')

        # weight initializer
        self.W_initializer = W_initializer
        self.bias_initializer = b_initializer
        is_initializer(W_initializer, 'weight_initializer')
        is_initializer(b_initializer, 'bias_initializer', allow_none=True)

        # parameter initialization
        W = parameter(self.W_initializer, (num_in, self.num_out))
        b = parameter(self.bias_initializer, (self.num_out,))
        if isinstance(self.mode, bm.TrainingMode):
            W = bm.TrainVar(W)
            b = None if (b is None) else bm.TrainVar(b)
        self.W = W
        self.b = b

        # fitting parameters
        self.online_fit_by = None  # support online training
        self.offline_fit_by = None  # support offline training
        self.fit_record = dict()

    def __repr__(self):
        return (f'{self.__class__.__name__}(name={self.name}, '
                f'num_in={self.num_in}, '
                f'num_out={self.num_out}, '
                f'mode={self.mode})')

    def update(self, x):
        x = bm.as_jax(x)
        res = x @ self.W
        if self.b is not None:
            res += self.b

        # online fitting data
        if share.load('fit', False) and self.online_fit_by is not None:
            self.fit_record['input'] = x
            self.fit_record['output'] = res

        # offline fitting data
        if share.load('fit', False) and self.offline_fit_by is not None:
            self.fit_record['input'] = x
            self.fit_record['output'] = res
        return res

    def online_init(self):
        if self.b is None:
            num_input = self.num_in
        else:
            num_input = self.num_in + 1
        self.online_fit_by.register_target(feature_in=num_input, identifier=self.name)

    def online_fit(self,
                   target: ArrayType,
                   fit_record: Dict[str, ArrayType]):
        if not isinstance(target, (bm.ndarray, jnp.ndarray)):
            raise MathError(f'"target" must be a tensor, but got {type(target)}')
        x = fit_record['input']
        y = fit_record['output']
        if x.ndim != 2:
            raise ValueError(f'"ff" must be a 2D tensor with shape of (num_sample, '
                             f'num_feature), but we got {x.shape}')
        if target.ndim != 2:
            raise ValueError(f'"target" must be a 2D tensor with shape of (num_sample, '
                             f'num_feature), but we got {target.shape}')
        if x.shape[0] != target.shape[0]:
            raise ValueError(f'Batch size of the input and target data should be '
                             f'the same, while we got {x.shape[0]} != {target.shape[0]}.')
        if target.shape[1] != y.shape[1]:
            raise MathError(f'The output dimension of output and target data should be '
                            f'the same, while we got {target.shape[1]} != {y.shape[1]}')

        # data
        if self.b is not None:
            x = jnp.concatenate([jnp.ones((x.shape[0], 1)), x], axis=-1)

        # fitting
        dW = self.online_fit_by.call(target=target, input=x, output=y, identifier=self.name)

        # assign trained weights
        if self.b is None:
            self.W += dW
        else:
            db, dW = jnp.split(dW, [1])
            self.b += db[0]
            self.W += dW

    def offline_fit(self,
                    target: ArrayType,
                    fit_record: Dict[str, ArrayType]):
        """The offline training interface for the Dense node."""
        # data checking
        if not isinstance(target, (bm.ndarray, jnp.ndarray)):
            raise MathError(f'"targets" must be a tensor, but got {type(target)}')
        xs = fit_record['input']
        ys = fit_record['output']
        if xs.ndim != 3:
            raise ValueError(f'"ffs" must be a 3D tensor with shape of (num_sample, num_time, '
                             f'num_feature), but we got {xs.shape}')
        if target.ndim != 3:
            raise ValueError(f'"targets" must be a 3D tensor with shape of (num_sample, num_time, '
                             f'num_feature), but we got {target.shape}')
        if ys.shape != target.shape:
            raise ValueError(f'The shapes of output and target data should be '
                             f'the same, while we got {ys.shape} != {target.shape}.')
        if xs.shape[0] != target.shape[0]:
            raise ValueError(f'Batch size of the input and target data should be '
                             f'the same, while we got {xs.shape[0]} != {target.shape[0]}.')
        if xs.shape[1] != target.shape[1]:
            raise MathError(f'The time dimension of input and target data should be '
                            f'the same, while we got {xs.shape[1]} != {target.shape[1]}')

        # get input and target training data
        if self.b is not None:
            xs = jnp.concatenate([jnp.ones(xs.shape[:2] + (1,)), xs], axis=-1)  # (..., 1 + num_ff_input)

        # solve weights by offline training methods
        weights = self.offline_fit_by(target, xs, ys)

        # assign trained weights
        if self.b is None:
            self.W.value = weights
        else:
            bias, Wff = jnp.split(weights, [1])
            self.W.value = Wff
            self.b.value = bias[0]

    def stdp_update(
        self,
        on_pre: Dict = None,
        on_post: Dict = None,
        w_min: numbers.Number = None,
        w_max: numbers.Number = None
    ):
        if not isinstance(self.W, bm.Variable):
            raise ValueError(f'When using STDP to update synaptic weights, the weight must be a variable.')
        if on_pre is not None:
            spike = on_pre['spike']
            trace = on_pre['trace']
            self.W.value = dense_on_pre(self.W.value, spike, trace, w_min, w_max)
        if on_post is not None:
            spike = on_post['spike']
            trace = on_post['trace']
            self.W.value = dense_on_post(self.W.value, trace, spike, w_min, w_max)


Linear = Dense


class Identity(Layer):
    r"""A placeholder identity operator that is argument-insensitive.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(Identity, self).__init__(*args, **kwargs)

    def update(self, x):
        return x


class AllToAll(Layer, SupportSTDP):
    """Synaptic matrix multiplication with All2All connections.

    Args:
      num_pre: int. The number of neurons in the presynaptic neuron group.
      num_post: int. The number of neurons in the postsynaptic neuron group.
      weight: The synaptic weights.
      sharding: The sharding strategy.
      include_self: bool. Whether connect the neuron with at the same position.
      mode: Mode. The computing mode.
      name: str. The object name.
    """

    def __init__(
        self,
        num_pre: int,
        num_post: int,
        weight: Union[float, ArrayType, Callable],
        sharding: Optional[Sharding] = None,
        include_self: bool = True,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super().__init__(mode=mode, name=name)

        self.num_pre = num_pre
        self.num_post = num_post
        self.include_self = include_self
        self.sharding = sharding

        weight = init.parameter(weight, (self.num_pre, self.num_post), sharding=sharding)
        if isinstance(self.mode, bm.TrainingMode):
            weight = bm.TrainVar(weight)
        self.weight = weight

    def update(self, pre_val):
        if bm.ndim(self.weight) == 0:  # weight is a scalar
            if isinstance(self.mode, bm.BatchingMode):
                assert pre_val.ndim == 2, 'Under the batching mode, the input should be a 2D array.'
                post_val = bm.sum(pre_val, keepdims=True, axis=1)
            else:
                assert pre_val.ndim == 1, 'Under the NonBatching mode, the input should be a 1D array.'
                post_val = bm.sum(pre_val)
            if not self.include_self:
                if self.num_pre == self.num_post:
                    post_val = post_val - pre_val
                elif self.num_pre > self.num_post:
                    val = pre_val[:self.num_post]
                    post_val = post_val - val
                else:
                    val = bm.concatenate([pre_val, bm.zeros(self.num_post - self.num_pre)])
                    post_val = post_val - val
            post_val = self.weight * post_val

        else:  # weight is a matrix
            assert self.weight.ndim == 2, '"weight" must be a 2D matrix.'
            if not self.include_self:
                post_val = pre_val @ bm.fill_diagonal(self.weight, 0., inplace=False)
            else:
                post_val = pre_val @ self.weight
        return post_val

    def stdp_update(
        self,
        on_pre: Dict = None,
        on_post: Dict = None,
        w_min: numbers.Number = None,
        w_max: numbers.Number = None
    ):
        if not isinstance(self.weight, bm.Variable):
            raise ValueError(f'When using STDP to update synaptic weights, the weight must be a variable.')
        if on_pre is not None:
            spike = on_pre['spike']
            trace = on_pre['trace']
            self.weight.value = dense_on_pre(self.weight.value, spike, trace, w_min, w_max)
        if on_post is not None:
            spike = on_post['spike']
            trace = on_post['trace']
            self.weight.value = dense_on_post(self.weight.value, trace, spike, w_min, w_max)


class OneToOne(Layer, SupportSTDP):
    """Synaptic matrix multiplication with One2One connection.

    Args:
      num: int. The number of neurons.
      weight: The synaptic weight.
      sharding: The sharding strategy.
      mode: The computing mode.
      name: The object name.

    """

    def __init__(
        self,
        num: int,
        weight: Union[float, ArrayType, Callable],
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super().__init__(mode=mode, name=name)

        self.num = num
        self.sharding = sharding

        weight = init.parameter(weight, (self.num,), sharding=sharding)
        if isinstance(self.mode, bm.TrainingMode):
            weight = bm.TrainVar(weight)
        self.weight = weight

    def update(self, pre_val):
        return pre_val * self.weight

    def stdp_update(
        self,
        on_pre: Dict = None,
        on_post: Dict = None,
        w_min: numbers.Number = None,
        w_max: numbers.Number = None
    ):
        if isinstance(self.weight, float):
            raise ValueError(f'Cannot update the weight of a constant node.')
        if not isinstance(self.weight, bm.Variable):
            self.tracing_variable('weight', self.weight, self.weight.shape)
        if on_pre is not None:
            spike = on_pre['spike']
            trace = on_pre['trace']
            self.weight.value += spike * trace
        if on_post is not None:
            spike = on_post['spike']
            trace = on_post['trace']
            self.weight.value += spike * trace


class MaskedLinear(Layer, SupportSTDP):
    r"""Synaptic matrix multiplication with masked dense computation.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
    :math:`M` the synaptic weight using a dense matrix.

    >>> import brainpy as bp
    >>> l = bp.dnn.MaskedLinear(bp.conn.FixedProb(0.1, pre=100, post=100),
    >>>                         weight=0.1)

    Args:
      conn: TwoEndConnector. The connection.
      weight: Synaptic weights. Can be a scalar, array, or callable function.
      mask_fun: Masking function.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        conn: connect.TwoEndConnector,
        weight: Union[float, ArrayType, Callable],
        mask_fun: Callable = Identity(),
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, mode=mode)

        assert isinstance(conn, connect.TwoEndConnector)
        self.conn = conn
        self.sharding = sharding
        self.mask_fun = mask_fun

        # weight
        weight = init.parameter(weight, (conn.pre_num, conn.post_num), sharding=sharding)
        if isinstance(self.mode, bm.TrainingMode):
            weight = bm.TrainVar(weight)
        self.weight = weight

        # connection
        self.mask = bm.sharding.partition(self.conn.require('conn_mat'), sharding=sharding)

    def update(self, x):
        return x @ self.mask_fun(self.weight * self.mask)

    def stdp_update(
        self,
        on_pre: Dict = None,
        on_post: Dict = None,
        w_min: numbers.Number = None,
        w_max: numbers.Number = None
    ):
        if isinstance(self.weight, float):
            raise ValueError(f'Cannot update the weight of a constant node.')
        if not isinstance(self.weight, bm.Variable):
            self.tracing_variable('weight', self.weight, self.weight.shape)
        if on_pre is not None:
            spike = on_pre['spike']
            trace = on_pre['trace']
            self.weight.value = dense_on_pre(self.weight.value, spike, trace, w_min, w_max)
        if on_post is not None:
            spike = on_post['spike']
            trace = on_post['trace']
            self.weight.value = dense_on_post(self.weight.value, trace, spike, w_min, w_max)


class _CSRLayer(Layer, SupportSTDP):
    def __init__(
        self,
        conn: connect.TwoEndConnector,
        weight: Union[float, ArrayType, Callable],
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        transpose: bool = True,
    ):
        super().__init__(name=name, mode=mode)

        assert isinstance(conn, connect.TwoEndConnector)
        assert sharding is None, 'Currently this model does not support sharding.'
        self.conn = conn
        self.sharding = sharding
        self.transpose = transpose

        # connection
        self.indices, self.indptr = self.conn.require('csr')

        # weight
        weight = init.parameter(weight, (self.indices.size,))
        if isinstance(self.mode, bm.TrainingMode):
            weight = bm.TrainVar(weight)
        self.weight = weight

    def stdp_update(
        self,
        on_pre: Dict = None,
        on_post: Dict = None,
        w_min: numbers.Number = None,
        w_max: numbers.Number = None
    ):
        if bm.isscalar(self.weight):
            raise ValueError(f'When using STDP to update synaptic weights, the weight cannot be a scalar.')
        if self.weight.shape != self.indices.shape:
            raise ValueError(
                f'The shape of weight should be the same as the shape of sparse weight {self.weight.shape}.')
        if not isinstance(self.weight, bm.Variable):
            self.tracing_variable('weight', self.weight, self.weight.shape)
        if on_pre is not None:  # update on presynaptic spike
            spike = on_pre['spike']
            trace = on_pre['trace']
            self.weight.value = csr_on_pre(
                self.weight.value, self.indices, self.indptr, spike, trace, w_min, w_max,
                shape=(spike.shape[0], trace.shape[0]),
            )
        if on_post is not None:  # update on postsynaptic spike
            if not hasattr(self, '_pre_ids'):
                with jax.ensure_compile_time_eval():
                    self._pre_ids, self._post_indptr, self.w_indices = csr2csc(
                        [self.indices, self.indptr], self.conn.post_num, data=np.arange(self.weight.size)
                    )
            spike = on_post['spike']
            trace = on_post['trace']
            self.weight.value = csr2csc_on_post(
                self.weight.value, self._pre_ids, self._post_indptr,
                self.w_indices, trace, spike, w_min, w_max,
                shape=(trace.shape[0], spike.shape[0]),
            )


class CSRLinear(_CSRLayer):
    r"""Synaptic matrix multiplication with CSR sparse computation.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
    :math:`M` the synaptic weight using a CSR sparse matrix.

    Args:
      conn: TwoEndConnector. The connection.
      weight: Synaptic weights. Can be a scalar, array, or callable function.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        conn: connect.TwoEndConnector,
        weight: Union[float, ArrayType, Callable],
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        method: str = None,
        transpose: bool = True,
    ):
        super().__init__(name=name, mode=mode, conn=conn, weight=weight, sharding=sharding, transpose=transpose)
        self.method = method

    def update(self, x):
        if x.ndim == 1:
            return bm.sparse.csrmv(self.weight, self.indices, self.indptr, x,
                                   shape=(self.conn.pre_num, self.conn.post_num), transpose=self.transpose)
        elif x.ndim > 1:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_csrmv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_csrmv(self, x):
        return bm.sparse.csrmv(self.weight, self.indices, self.indptr, x,
                               shape=(self.conn.pre_num, self.conn.post_num), transpose=self.transpose)


class EventCSRLinear(_CSRLayer):
    r"""Synaptic matrix multiplication with event CSR sparse computation.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic spikes,
    :math:`M` the synaptic weight using a CSR sparse matrix.

    Args:
      conn: TwoEndConnector. The connection.
      weight: Synaptic weights. Can be a scalar, array, or callable function.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        conn: connect.TwoEndConnector,
        weight: Union[float, ArrayType, Callable],
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        transpose: bool = True,
    ):
        super().__init__(name=name, mode=mode, conn=conn, weight=weight, sharding=sharding, transpose=transpose)

    def update(self, x):
        if x.ndim == 1:
            return bm.event.csrmv(self.weight, self.indices, self.indptr, x,
                                  shape=(self.conn.pre_num, self.conn.post_num),
                                  transpose=self.transpose)
        elif x.ndim > 1:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_csrmv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_csrmv(self, x):
        return bm.event.csrmv(self.weight, self.indices, self.indptr, x,
                              shape=(self.conn.pre_num, self.conn.post_num),
                              transpose=self.transpose)


class CSCLinear(Layer):
    r"""Synaptic matrix multiplication with CSC sparse computation.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
    :math:`M` the synaptic weight using a CSC sparse matrix.

    Args:
      conn: TwoEndConnector. The connection.
      weight: Synaptic weights. Can be a scalar, array, or callable function.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        conn: connect.TwoEndConnector,
        weight: Union[float, ArrayType, Callable],
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, mode=mode)

        assert isinstance(conn, connect.TwoEndConnector)
        self.conn = conn
        self.sharding = sharding


class BcsrMM(Layer):
    r"""Synaptic matrix multiplication with BCSR sparse computation.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
    :math:`M` the synaptic weight using a BCSR sparse matrix.

    Args:
      conn: TwoEndConnector. The connection.
      weight: Synaptic weights. Can be a scalar, array, or callable function.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        conn: connect.TwoEndConnector,
        weight: Union[float, ArrayType, Callable],
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, mode=mode)

        assert isinstance(conn, connect.TwoEndConnector)
        self.conn = conn
        self.sharding = sharding


class BcscMM(Layer):
    r"""Synaptic matrix multiplication with BCSC sparse computation.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
    :math:`M` the synaptic weight using a BCSC sparse matrix.

    Args:
      conn: TwoEndConnector. The connection.
      weight: Synaptic weights. Can be a scalar, array, or callable function.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        conn: connect.TwoEndConnector,
        weight: Union[float, ArrayType, Callable],
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, mode=mode)

        assert isinstance(conn, connect.TwoEndConnector)
        self.conn = conn
        self.sharding = sharding


class JitLinear(Layer):
    def get_conn_matrix(self):
        pass


class JitFPHomoLayer(JitLinear):
    def get_conn_matrix(self):
        return bm.jitconn.get_homo_weight_matrix(self.weight, self.prob, self.seed,
                                                 shape=(self.num_out, self.num_in),
                                                 transpose=self.transpose,
                                                 outdim_parallel=not self.atomic)


class JitFPUniformLayer(JitLinear):
    def get_conn_matrix(self):
        return bm.jitconn.get_uniform_weight_matrix(self.w_low, self.w_high, self.prob, self.seed,
                                                    shape=(self.num_out, self.num_in),
                                                    transpose=self.transpose,
                                                    outdim_parallel=not self.atomic)


class JitFPNormalLayer(JitLinear):
    def get_conn_matrix(self):
        return bm.jitconn.get_normal_weight_matrix(self.w_mu, self.w_sigma, self.prob, self.seed,
                                                   shape=(self.num_out, self.num_in),
                                                   transpose=self.transpose,
                                                   outdim_parallel=not self.atomic)


class JitFPHomoLinear(JitFPHomoLayer):
    r"""Synaptic matrix multiplication with the just-in-time connectivity.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic variable,
    :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
    Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
    and at each connection, the synaptic value is the same :math:`weight`.

    Args:
      num_in: int. The number of the input feature. A positive integer.
      num_out: int. The number of the input feature. A positive integer.
      prob: float. The connectivity probability.
      weight: float. The synaptic value at each position.
      seed: int. The random seed used to keep the reproducibility of the connectivity.
      transpose: bool. Transpose the JIT matrix or not. Default False.
      atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
         May be changed in the future.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        prob: float,
        weight: float,
        seed: Optional[int] = None,
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        transpose: bool = False,
        atomic: bool = False,
    ):
        super().__init__(name=name, mode=mode)

        self.prob = prob
        self.sharding = sharding
        self.transpose = transpose
        self.seed = np.random.randint(0, 100000) if seed is None else seed
        self.atomic = atomic
        self.num_in = num_in
        self.num_out = num_out

        # weight
        if isinstance(self.mode, bm.TrainingMode):
            weight = bm.TrainVar(weight)
        self.weight = weight

    def update(self, x):
        if x.ndim == 1:
            return bm.jitconn.mv_prob_homo(x, self.weight, self.prob, self.seed,
                                           shape=(self.num_out, self.num_in),
                                           transpose=self.transpose,
                                           outdim_parallel=not self.atomic)
        elif x.ndim == 2:
            return jax.vmap(self._batch_mv)(x)
        elif x.ndim > 2:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_mv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_mv(self, x):
        return bm.jitconn.mv_prob_homo(x, self.weight, self.prob, self.seed,
                                       shape=(self.num_out, self.num_in),
                                       transpose=self.transpose,
                                       outdim_parallel=not self.atomic)


class JitFPUniformLinear(JitFPUniformLayer):
    r"""Synaptic matrix multiplication with the just-in-time connectivity.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic variable,
    :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
    Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
    and at each connection, the synaptic value is sample from a uniform distribution :math:`U(w_{low}, w_{high})`.

    Args:
      num_in: int. The number of the input feature. A positive integer.
      num_out: int. The number of the input feature. A positive integer.
      prob: float. The connectivity probability.
      w_low: float. The lowest value of the uniform distribution.
      w_high: float. The highest value of the uniform distribution.
      seed: int. The random seed used to keep the reproducibility of the connectivity.
      transpose: bool. Transpose the JIT matrix or not. Default False.
      atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
         May be changed in the future.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        prob: float,
        w_low: float,
        w_high: float,
        seed: Optional[int] = None,
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        transpose: bool = False,
        atomic: bool = False,
    ):
        super().__init__(name=name, mode=mode)

        self.prob = prob
        self.sharding = sharding
        self.transpose = transpose
        self.seed = np.random.randint(0, 100000) if seed is None else seed
        self.atomic = atomic
        self.num_in = num_in
        self.num_out = num_out

        # weight
        self.w_low = w_low
        self.w_high = w_high

    def update(self, x):
        if x.ndim == 1:
            return bm.jitconn.mv_prob_uniform(x, self.w_low, self.w_high, self.prob, self.seed,
                                              shape=(self.num_out, self.num_in),
                                              transpose=self.transpose,
                                              outdim_parallel=not self.atomic)
        elif x.ndim == 2:
            return jax.vmap(self._batch_mv)(x)
        elif x.ndim > 2:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_mv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_mv(self, x):
        return bm.jitconn.mv_prob_uniform(x, self.w_low, self.w_high, self.prob, self.seed,
                                          shape=(self.num_out, self.num_in),
                                          transpose=self.transpose,
                                          outdim_parallel=not self.atomic)


class JitFPNormalLinear(JitFPNormalLayer):
    r"""Synaptic matrix multiplication with the just-in-time connectivity.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic variable,
    :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
    Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
    and at each connection, the synaptic value is sample from a normal distribution :math:`N(\mu, \sigma)`.

    Args:
      num_in: int. The number of the input feature. A positive integer.
      num_out: int. The number of the input feature. A positive integer.
      prob: float. The connectivity probability.
      w_mu: float. The center of the normal distribution.
      w_sigma: float. The standard variance of the normal distribution.
      seed: int. The random seed used to keep the reproducibility of the connectivity.
      transpose: bool. Transpose the JIT matrix or not. Default False.
      atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
         May be changed in the future.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        prob: float,
        w_mu: float,
        w_sigma: float,
        seed: Optional[int] = None,
        sharding: Optional[Sharding] = None,
        transpose: bool = False,
        atomic: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, mode=mode)

        self.prob = prob
        self.sharding = sharding
        self.transpose = transpose
        self.seed = np.random.randint(0, 100000) if seed is None else seed
        self.atomic = atomic
        self.num_in = num_in
        self.num_out = num_out

        # weight
        self.w_mu = w_mu
        self.w_sigma = w_sigma

    def update(self, x):
        if x.ndim == 1:
            return bm.jitconn.mv_prob_normal(x, self.w_mu, self.w_sigma, self.prob, self.seed,
                                             shape=(self.num_out, self.num_in),
                                             transpose=self.transpose,
                                             outdim_parallel=not self.atomic)
        elif x.ndim == 2:
            return jax.vmap(self._batch_mv)(x)
        elif x.ndim > 2:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_mv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_mv(self, x):
        return bm.jitconn.mv_prob_normal(x, self.w_mu, self.w_sigma, self.prob, self.seed,
                                         shape=(self.num_out, self.num_in),
                                         transpose=self.transpose,
                                         outdim_parallel=not self.atomic)


class EventJitFPHomoLinear(JitFPHomoLayer):
    r"""Synaptic matrix multiplication with the just-in-time connectivity.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic spikes,
    :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
    Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
    and at each connection, the synaptic value is the same :math:`weight`.

    Args:
      num_in: int. The number of the input feature. A positive integer.
      num_out: int. The number of the input feature. A positive integer.
      prob: float. The connectivity probability.
      weight: float. The synaptic value at each position.
      seed: int. The random seed used to keep the reproducibility of the connectivity.
      transpose: bool. Transpose the JIT matrix or not. Default False.
      atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
         May be changed in the future.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        prob: float,
        weight: float,
        seed: Optional[int] = None,
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        transpose: bool = False,
        atomic: bool = True,
    ):
        super().__init__(name=name, mode=mode)

        self.prob = prob
        self.sharding = sharding
        self.transpose = transpose
        self.seed = np.random.randint(0, 1000000) if seed is None else seed
        self.atomic = atomic
        self.num_in = num_in
        self.num_out = num_out

        # weight
        if isinstance(self.mode, bm.TrainingMode):
            weight = bm.TrainVar(weight)
        self.weight = weight

    def update(self, x):
        if x.ndim == 1:
            return bm.jitconn.event_mv_prob_homo(x, self.weight, self.prob, self.seed,
                                                 shape=(self.num_out, self.num_in),
                                                 transpose=self.transpose,
                                                 outdim_parallel=not self.atomic)
        elif x.ndim == 2:
            return jax.vmap(self._batch_mv)(x)
        elif x.ndim > 2:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_mv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_mv(self, x):
        return bm.jitconn.event_mv_prob_homo(x, self.weight, self.prob, self.seed,
                                             shape=(self.num_out, self.num_in),
                                             transpose=self.transpose,
                                             outdim_parallel=not self.atomic)


class EventJitFPUniformLinear(JitFPUniformLayer):
    r"""Synaptic matrix multiplication with the just-in-time connectivity.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic spikes,
    :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
    Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
    and at each connection, the synaptic value is sample from a uniform distribution :math:`U(w_{low}, w_{high})`.

    Args:
      num_in: int. The number of the input feature. A positive integer.
      num_out: int. The number of the input feature. A positive integer.
      prob: float. The connectivity probability.
      w_low: float. The lowest value of the uniform distribution.
      w_high: float. The highest value of the uniform distribution.
      seed: int. The random seed used to keep the reproducibility of the connectivity.
      transpose: bool. Transpose the JIT matrix or not. Default False.
      atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
         May be changed in the future.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        prob: float,
        w_low: float,
        w_high: float,
        seed: Optional[int] = None,
        sharding: Optional[Sharding] = None,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        transpose: bool = False,
        atomic: bool = True,
    ):
        super().__init__(name=name, mode=mode)

        self.prob = prob
        self.sharding = sharding
        self.transpose = transpose
        self.seed = np.random.randint(0, 100000) if seed is None else seed
        self.atomic = atomic
        self.num_in = num_in
        self.num_out = num_out

        # weight
        self.w_low = w_low
        self.w_high = w_high

    def update(self, x):
        if x.ndim == 1:
            return bm.jitconn.event_mv_prob_uniform(x, self.w_low, self.w_high, self.prob, self.seed,
                                                    shape=(self.num_out, self.num_in),
                                                    transpose=self.transpose,
                                                    outdim_parallel=not self.atomic)
        elif x.ndim == 2:
            return jax.vmap(self._batch_mv)(x)
        elif x.ndim > 2:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_mv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_mv(self, x):
        return bm.jitconn.event_mv_prob_uniform(x, self.w_low, self.w_high, self.prob, self.seed,
                                                shape=(self.num_out, self.num_in),
                                                transpose=self.transpose,
                                                outdim_parallel=not self.atomic)


class EventJitFPNormalLinear(JitFPNormalLayer):
    r"""Synaptic matrix multiplication with the just-in-time connectivity.

    It performs the computation of:

    .. math::

       y = x @ M

    where :math:`y` is the postsynaptic value, :math:`x` the presynaptic spikes,
    :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
    Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
    and at each connection, the synaptic value is sample from a normal distribution :math:`N(\mu, \sigma)`.

    Args:
      num_in: int. The number of the input feature. A positive integer.
      num_out: int. The number of the input feature. A positive integer.
      prob: float. The connectivity probability.
      w_mu: float. The center of the normal distribution.
      w_sigma: float. The standard variance of the normal distribution.
      seed: int. The random seed used to keep the reproducibility of the connectivity.
      transpose: bool. Transpose the JIT matrix or not. Default False.
      atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
         May be changed in the future.
      sharding: The sharding strategy.
      mode: The synaptic computing mode.
      name: The synapse model name.
    """

    def __init__(
        self,
        num_in: int,
        num_out: int,
        prob: float,
        w_mu: float,
        w_sigma: float,
        seed: Optional[int] = None,
        sharding: Optional[Sharding] = None,
        transpose: bool = False,
        atomic: bool = True,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, mode=mode)

        self.prob = prob
        self.sharding = sharding
        self.transpose = transpose
        self.seed = np.random.randint(0, 100000) if seed is None else seed
        self.atomic = atomic
        self.num_in = num_in
        self.num_out = num_out

        # weight
        self.w_mu = w_mu
        self.w_sigma = w_sigma

    def update(self, x):
        if x.ndim == 1:
            return bm.jitconn.event_mv_prob_normal(x, self.w_mu, self.w_sigma, self.prob, self.seed,
                                                   shape=(self.num_out, self.num_in),
                                                   transpose=self.transpose,
                                                   outdim_parallel=not self.atomic)
        elif x.ndim == 2:
            return jax.vmap(self._batch_mv)(x)
        elif x.ndim > 2:
            shapes = x.shape[:-1]
            x = bm.flatten(x, end_dim=-2)
            y = jax.vmap(self._batch_mv)(x)
            return bm.reshape(y, shapes + (y.shape[-1],))
        else:
            raise ValueError

    def _batch_mv(self, x):
        return bm.jitconn.event_mv_prob_normal(x, self.w_mu, self.w_sigma, self.prob, self.seed,
                                               shape=(self.num_out, self.num_in),
                                               transpose=self.transpose,
                                               outdim_parallel=not self.atomic)
