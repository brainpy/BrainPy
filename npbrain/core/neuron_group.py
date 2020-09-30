# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import OrderedDict

from .types import ObjState
from .base import BaseType
from .base import BaseEnsemble
from .. import _numpy as np
from .. import profile
from ..utils import helper

__all__ = [
    'NeuType',
    'NeuGroup',
]

_neu_no = 0


class NeuType(BaseType):
    """Abstract Neuron Type.

    It can be defined based on a group of neurons or a single neuron.
    """

    def __init__(self, name, create_func, group_based=True):
        super(NeuType, self).__init__(create_func=create_func, name=name, group_based=group_based, type_='neu')


class NeuGroup(BaseEnsemble):
    """Neuron Group.
    """

    def __init__(self, model, geometry, monitors=None, vars_init=None, pars_update=None, name=None):
        # model
        # ------
        assert isinstance(model, NeuType), 'Must provide an instance of NeuType class.'

        # name
        # -----
        if name is None:
            global _neu_no
            name = f'NeuGroup{_neu_no}'
            _neu_no += 1
        else:
            name = name

        # num and geometry
        # -----------------
        if isinstance(geometry, (int, float)):
            geometry = (1, int(geometry))
        elif isinstance(geometry, (tuple, list)):
            if len(geometry) == 1:
                height, width = 1, geometry[0]
            elif len(geometry) == 2:
                height, width = geometry[0], geometry[1]
            else:
                raise ValueError('Do not support 3+ dimensional networks.')
            geometry = (height, width)
        else:
            raise ValueError()
        num = int(np.prod(geometry))
        self.geometry = geometry

        # initialize
        # ----------
        super(NeuGroup, self).__init__(model=model, name=name, num=num, pars_update=pars_update,
                                       vars_init=vars_init, monitors=monitors)

        # ST
        # --
        self.ST = ObjState(self.vars_init)(num)

        # model update schedule
        # ---------------------
        self._schedule = ['input', 'step_func', 'monitor']

    @property
    def _keywords(self):
        kws = ['model', 'num', 'geometry', 'ST', 'PA',
               'vars_init', 'pars_update',
               'mon', '_mon_vars', 'step_func', '_schedule']
        if hasattr(self, 'model'):
            return kws + self.model.step_names
        else:
            return kws

    def __setattr__(self, key, value):
        if key in self._keywords:
            if hasattr(self, key):
                raise KeyError(f'"{key}" is a keyword in NeuGroup, please change another name.')
        super(NeuGroup, self).__setattr__(key, value)

    def _add_input(self, key_val_ops_types):
        code_scope = {self.name: self}
        code_args, code_arg2call = set(), {}
        code_lines = []

        # check datatype of the input
        # ----------------------------
        has_iter = False
        for _, _, _, t in key_val_ops_types:
            assert t in ['iter', 'fix'], 'Only support inputs of "iter" and "fix" types.'
            if t == 'iter':
                has_iter = True
        if has_iter:
            code_args.add('i')
            code_arg2call['i'] = 'i'

        # check data operations
        # ----------------------
        for _, _, ops, _ in key_val_ops_types:
            assert ops in ['-', '+', 'x', '/', '='], 'Only support five operations: +, -, x, /, ='
        ops2str = {'-': 'sub', '+': 'add', 'x': 'mul', '/': 'div', '=': 'assign'}

        # generate code of variable input
        # --------------------------------
        input_idx = 0
        for key, val, ops, data_type in key_val_ops_types:
            attr_item = key.split('.')

            # get the left side #
            if len(attr_item) == 1 and (attr_item[0] not in self.ST):  # if "item" is the model attribute
                attr, item = attr_item[0], ''
                assert hasattr(self, attr), f'Model "{self.name}" doesn\'t have "{attr}" attribute", ' \
                                            f'and "{self.name}.ST" doesn\'t have "{attr}" field.'
                assert isinstance(getattr(self, attr), np.ndarray), f'NumpyBrain only support input to arrays.'

                if profile.is_numpy_bk():
                    left = f'{self.name}.{attr}'
                else:
                    left = f'{self.name}_{attr}'
                    code_args.add(left)
                    code_arg2call[left] = f'{self.name}.{attr}'
            else:
                if len(attr_item) == 1:
                    attr, item = 'ST', attr_item[0]
                elif len(attr_item) == 2:
                    attr, item = attr_item[0], attr_item[1]
                else:
                    raise ValueError(f'Unknown target : {key}.')
                assert item in getattr(self, attr), f'"{self.name}.{attr}" doesn\'t have "{item}" field.'

                if profile.is_numpy_bk():
                    left = f'{self.name}.{attr}["{item}"]'
                else:
                    idx = getattr(self, attr)['_var2idx'][item]
                    left = f'{self.name}_{attr}{idx}]'
                    code_args.add(f'{self.name}_{attr}')
                    code_arg2call[f'{self.name}_{attr}'] = f'{self.name}.{attr}["_data"]'

            # get the right side #
            right = f'{self.name}_input{input_idx}_{attr}_{item}_{ops2str[ops]}'
            code_scope[right] = val
            if data_type == 'iter':
                right = right + '[i]'
            input_idx += 1

            # final code line #
            if ops == '=':
                code_lines.append(left + " = " + right)
            else:
                code_lines.append(left + f" {ops}= " + right)

        from pprint import pprint

        print("code_scope: ")
        pprint(code_scope)
        print("code_args: ")
        pprint(code_args)
        print("code_arg2call: ")
        pprint(code_arg2call)
        pprint('\n'.join(code_lines))
