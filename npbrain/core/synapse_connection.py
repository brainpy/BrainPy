# -*- coding: utf-8 -*-

from .base import BaseEnsemble
from .base import BaseType
from .connectivity import Connector
from .connectivity import post2syn
from .connectivity import pre2syn
from .neuron_group import NeuGroup
from .types import ObjState
from .. import _numpy as np
from .. import profile

__all__ = [
    'SynType',
    'SynConn',
]

_syn_no = 0


class SynType(BaseType):
    """Abstract Synapse Type.

    It can be defined based on a collection of synapses or a single synapse model.
    """

    def __init__(self, create_func, group_based=True, name=None):
        super(SynType, self).__init__(create_func=create_func, name=name, group_based=group_based, type_='syn')


class SynConn(BaseEnsemble):
    """Synaptic connections.

    """

    def __init__(self, model, delay=0.,
                 pre_group=None, post_group=None, conn=None,
                 num=None,
                 monitors=None, vars_init=None, pars_update=None, name=None):
        assert isinstance(model, SynType), 'Must provide an instance of SynType class.'

        # name
        # ----
        if name is None:
            global _syn_no
            name = f'SynConn{_syn_no}'
            _syn_no += 1
        else:
            name = name

        # pre or post neuron group
        # ------------------------
        if pre_group is not None and post_group is not None:
            # check
            # ------
            assert isinstance(pre_group, NeuGroup), '"pre" must be an instance of NeuGroup.'
            assert isinstance(post_group, NeuGroup), '"post" must be an instance of NeuGroup.'

            # connections
            # ------------
            if conn is None:
                if isinstance(conn, Connector):
                    pre_idx, post_idx = conn(pre_group.geometry, post_group.geometry)
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

            else:
                assert num is not None, '"num" must be provided when "conn" are none.'
                assert 0 < num, '"num" must be a positive number.'
                num = num

            # essential
            # ---------
            self.pre_group = pre_group
            self.post_group = post_group

        else:
            assert num is not None, '"num" must be provided when "pre" and "post" are none.'
            assert 0 < num, '"num" must be a positive number.'
            num = num

        # delay
        # -------
        if delay is None:
            delay_len = 1
        elif isinstance(delay, (int, float)):
            dt = profile.get_dt()
            delay_len = int(np.ceil(delay / dt)) + 1
        else:
            raise ValueError("NumpyBrain currently doesn't support other kinds of delay.")
        self.dlen = delay_len  # delay length
        self.din = 0  # delay in position
        self.dout = self.dlen - 1  # delay out position

        # initialize
        # ----------
        super(SynConn, self).__init__(model=model, name=name, num=num, pars_update=pars_update,
                                      vars_init=vars_init, monitors=monitors)

        # ST
        # --
        self.ST = ObjState(self.vars_init)((delay_len, self.num))

        self.step_funcs.append(self.update_delay_indices)

        # model update schedule
        # ---------------------
        self._schedule = ['step_funcs', 'monitor', 'delay_index']

    def update_delay_indices(self):
        # in_index
        self.din = (self.din + 1) % self.dlen
        # out_index
        self.dout = (self.dout + 1) % self.dlen

    @property
    def _keywords(self):
        kws = ['model', 'num', 'dlen', 'din', 'dout', 'ST', 'PA',
               'vars_init', 'pars_update',
               'mon', '_mon_vars', 'step_funcs', '_schedule']
        if hasattr(self, 'model'):
            return kws + self.model.step_names
        else:
            return kws

    def __setattr__(self, key, value):
        if key in self._keywords:
            if hasattr(self, key):
                raise KeyError(f'"{key}" is a keyword in SynConn, please change another name.')
        super(SynConn, self).__setattr__(key, value)
