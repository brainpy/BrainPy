# -*- coding: utf-8 -*-

import re

from .base_objects import BaseEnsemble
from .base_objects import BaseType
from .base_objects import ModelDefError
from .base_objects import ModelUseError
from .constants import _SYN_CONN
from .neurons import NeuGroup
from .types import SynState
from .. import numpy as np
from .. import profile
from .. import tools
from ..connectivity import Connector
from ..connectivity import post2syn
from ..connectivity import pre2syn

__all__ = [
    'SynType',
    'SynConn',
    'delayed',
]

_SYN_CONN_NO = 0


class SynType(BaseType):
    """Abstract Synapse Type.

    It can be defined based on a collection of synapses or a single synapse model.
    """

    def __init__(self, name, requires, steps, vector_based=True, heter_params_replace=None):
        super(SynType, self).__init__(requires=requires, steps=steps, name=name, vector_based=vector_based,
                                      heter_params_replace=heter_params_replace)

        # inspect delay keys
        # ------------------

        # delay function
        delay_funcs = []
        for func in self.steps:
            if func.__name__.startswith('_npbrain_delayed_'):
                delay_funcs.append(func)
        if len(delay_funcs):
            delay_func_code = '\n'.join([tools.deindent(tools.get_main_code(func)) for func in delay_funcs])
            delay_func_code_left = '\n'.join(tools.format_code(delay_func_code).lefts)

            # get delayed variables
            _delay_keys = dict()
            for arg, state in self.requires.items():
                if isinstance(state, SynState):
                    delay_keys_in_left = set(re.findall(r'' + arg + r'\[[\'"](\w+)[\'"]\]', delay_func_code_left))
                    if len(delay_keys_in_left) > 0:
                        raise ModelDefError(f'Delayed function cannot assign value to "{arg}".')
                    delay_keys = set(re.findall(r'' + arg + r'\[[\'"](\w+)[\'"]\]', delay_func_code))
                    if len(delay_keys) > 0:
                        if arg not in _delay_keys:
                            _delay_keys[arg] = set()
                        _delay_keys[arg].update(delay_keys)
            # self._delay_keys = _delay_keys.get('ST', set())
            self._delay_keys = _delay_keys

    def run(self):
        raise NotImplementedError


class SynConn(BaseEnsemble):
    """Synaptic connections.

    Parameters
    ----------
    model : NeuType
        The instantiated neuron type model.
    pars_update : dict, None
        Parameters to update.
    pre_group : NeuGroup, None
        Pre-synaptic neuron group.
    post_group : NeuGroup, None
        Post-synaptic neuron group.
    conn : Connector, None
        Connection method to create synaptic connectivity.
    num : int
        The number of the synapses.
    delay : float
        The time of the synaptic delay.
    monitors : list, tuple, None
        Variables to monitor.
    name : str, None
        The name of the neuron group.
    """

    def __init__(self, model, pars_update=None,
                 pre_group=None, post_group=None, conn=None,
                 num=None,
                 delay=0., monitors=None, name=None):
        # name
        # ----
        if name is None:
            global _SYN_CONN_NO
            name = f'SC{_SYN_CONN_NO}'
            _SYN_CONN_NO += 1
        else:
            name = name

        # pre or post neuron group
        # ------------------------
        self.pre_group = pre_group
        self.post_group = post_group
        if pre_group is not None and post_group is not None:
            # check
            # ------
            assert isinstance(pre_group, NeuGroup), '"pre" must be an instance of NeuGroup.'
            assert isinstance(post_group, NeuGroup), '"post" must be an instance of NeuGroup.'
            assert conn is not None, '"conn" must be provided.'

            # connections
            # ------------
            if isinstance(conn, Connector):
                conn_res = conn(pre_group.geometry, post_group.geometry)
                pre_idx, post_idx = conn_res['i'], conn_res['j']
            elif isinstance(conn, np.ndarray):
                assert np.ndim(conn) == 2, f'"conn" must be a 2D array, not {np.ndim(conn)}D.'
                conn_shape = np.shape(conn)
                assert conn_shape[0] == pre_group.num and conn_shape[1] == post_group.num, \
                    f'The shape of "conn" must be ({pre_group.num}, {post_group.num})'
                pre_idx, post_idx = [], []
                for i in enumerate(pre_group.num):
                    idx = np.where(conn[i] > 0)[0]
                    pre_idx.extend([i * len(idx)])
                    post_idx.extend(idx)
                pre_idx = np.asarray(pre_idx, dtype=np.int_)
                post_idx = np.asarray(post_idx, dtype=np.int_)
            else:
                assert isinstance(conn, dict), '"conn" only support "dict" or a 2D "array".'
                assert 'i' in conn, '"conn" must provide "i" item.'
                assert 'j' in conn, '"conn" must provide "j" item.'
                pre_idx = np.asarray(conn['i'], dtype=np.int_)
                post_idx = np.asarray(conn['j'], dtype=np.int_)

            num = len(pre_idx)
            self.pre2syn = pre2syn(pre_idx, post_idx, pre_group.num)
            self.post2syn = post2syn(pre_idx, post_idx, post_group.num)
            self.pre_ids = pre_idx
            self.post_ids = post_idx
            self.pre = pre_group.ST
            self.post = post_group.ST

        else:
            assert num is not None, '"num" must be provided when "pre" and "post" are none.'
        assert 0 < num < 2 ** 64, 'Total synapse number "num" must be a valid number in "uint64".'

        # delay
        # -------
        if delay is None:
            delay_len = 0
        elif isinstance(delay, (int, float)):
            dt = profile.get_dt()
            delay_len = int(np.ceil(delay / dt))
        else:
            raise ValueError("BrainPy currently doesn't support other kinds of delay.")
        self.delay_len = delay_len  # delay length

        # model
        # ------
        try:
            assert isinstance(model, SynType)
        except AssertionError:
            raise ModelUseError(f'{type(self).__name__} receives an instance of {SynType.__name__}, '
                                f'not {type(model).__name__}.')

        if not model.vector_based:
            if self.pre_group is None or self.post_group is None:
                raise ModelUseError('Using of scalar-based synapse model must '
                                    'provide "pre_group" and "post_group".')

        # initialize
        # ----------
        super(SynConn, self).__init__(model=model,
                                      pars_update=pars_update,
                                      name=name, num=num,
                                      monitors=monitors, cls_type=_SYN_CONN)

        # ST
        # --
        self.ST = self.requires['ST'].make_copy(size=self.num, delay=delay_len,
                                                delay_vars=list(self.model._delay_keys['ST']))

    def _merge_steps(self):
        codes_of_calls = super(SynConn, self)._merge_steps()
        codes_of_calls.append(f'{self.name}.ST._update_delay_indices()')
        return codes_of_calls

    @property
    def _keywords(self):
        return super(SynConn, self)._keywords + ['delay_len']


def post_cond_by_post2syn(syn_val, post2syn):
    num_post = len(post2syn)
    g_val = np.zeros(num_post, dtype=np.float_)
    for i in range(num_post):
        syn_idx = post2syn[i]
        g_val[i] = syn_val[syn_idx]
    return g_val


def delayed(func):
    func.__name__ = f'_npbrain_delayed_{func.__name__}'
    return func