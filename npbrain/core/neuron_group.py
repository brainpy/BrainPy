# -*- coding: utf-8 -*-

from .base import BaseEnsemble
from .base import BaseType
from .types import ObjState
from .. import _numpy as np
from .. import profile

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
                                       vars_init=vars_init, monitors=monitors, cls_type='neu_group')

        # ST
        # --
        self.ST = ObjState(self.vars_init)(num)

        # model update schedule
        # ---------------------
        self._schedule = ['input', 'step_func', 'monitor']

    @property
    def _keywords(self):
        return super(NeuGroup, self)._keywords + ['geometry', ]

    def __setattr__(self, key, value):
        if key in self._keywords:
            if hasattr(self, key):
                raise KeyError(f'"{key}" is a keyword in NeuGroup, please change another name.')
        super(NeuGroup, self).__setattr__(key, value)

    def _add_input(self, key_val_ops_types):
        code_scope, code_args, code_arg2call, code_lines = {}, set(), {}, []
        input_idx = 0

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

        # generate code of input function
        # --------------------------------
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

        # final code
        # ----------
        code_lines.insert(0, f'# Input step function of {self.name}')

        if profile.is_numpy_bk():
            code_args = list(code_args)
            code_lines.insert(0, f'\ndef input_step({", ".join(code_args)})')

            # compile function
            exec(compile('\n\t'.join(code_lines), '', 'exec'), code_scope)
            self.input_step = code_scope['input_step']

            # format function call
            code_arg2call = [code_arg2call[arg] for arg in code_args]
            func_call = f'{self.name}.input_step({", ".join(code_arg2call)})'

            if profile.show_codgen:
                print("\n" + '\n\t'.join(code_lines))
                print("\n" + func_call)

            self._codegen['input'] = {'funcs': self.input_step, 'calls': func_call}

        else:
            self._codegen['input'] = {'scopes': code_scope, 'args': code_args,
                                      'arg2calls': code_arg2call, 'codes': code_lines}

    def _merge_steps(self):
        codes_of_calls = []  # call the compiled functions

        self._type_checking()

        lines, scopes, args, arg2calls = [], dict(), set(), dict()
        for item in self._schedule:
            if profile.is_numpy_bk():
                if item in ['input', 'monitor']:
                    codes_of_calls.append(self._codegen[item]['calls'])
                else:
                    codes_of_calls.extend(self._codegen[item]['calls'])
            else:
                lines.extend(self._codegen[item]['codes'])
                scopes.update(self._codegen[item]['scopes'])
                args = args | self._codegen[item]['args']
                arg2calls.update(self._codegen[item]['arg2calls'])

        if not profile.is_numpy_bk():
            args = list(args)
            arg2calls_list = [arg2calls[arg] for arg in args]
            lines.insert(0, f'\ndef merge_func({", ".join(args)})')
            exec(compile('\n  '.join(lines), '', 'exec'), scopes)

            self.merge_func = scopes['merge_func']
            call = f'{self.name}.merge_func({", ".join(arg2calls_list)})'
            codes_of_calls.append(call)

            if profile.show_codgen:
                print("\n" + '\n\t'.join(lines))
                print("\n" + call)

        return codes_of_calls

    def set_schedule(self, schedule):
        assert isinstance(schedule, (list, tuple)), '"schedule" must be a list/tuple.'
        for s in schedule:
            assert s in ['input', 'monitor', 'step_func'], 'Use can only schedule "input", "monitor" and "step_func".'
        super(NeuGroup, self).__setattr__('_schedule', schedule)
