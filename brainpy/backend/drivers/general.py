# -*- coding: utf-8 -*-

from pprint import pprint

from brainpy import backend
from brainpy import errors
from brainpy.backend import ops
from brainpy.simulation import drivers
from . import utils

__all__ = [
    'GeneralDiffIntDriver',
    'GeneralNodeDriver',
    'GeneralNetDriver',
]


class GeneralDiffIntDriver(drivers.BaseDiffIntDriver):
    def build(self, *args, **kwargs):
        # compile
        code = '\n'.join(self.code_lines)
        if self.show_code:
            print(code)
            print()
            pprint(self.code_scope)
            print()
        exec(compile(code, '', 'exec'), self.code_scope)

        # attribute assignment
        new_f = self.code_scope[self.func_name]
        for key, value in self.uploads.items():
            setattr(new_f, key, value)
        return new_f


class GeneralNodeDriver(drivers.BaseNodeDriver):
    """General BrainPy Node Running Driver for NumPy, PyTorch, TensorFlow, etc.
    """

    def __init__(self, pop, steps=None):
        steps = pop.steps if steps is None else pop.steps
        super(GeneralNodeDriver, self).__init__(host=pop, steps=steps)
        self.last_inputs = {}
        self.formatted_funcs = {}
        self.run_func = None

    def _check_inputs_change(self, formatted_inputs, show_code):
        # check whether the input is changed
        # --
        new_inputs = {}
        input_keep_same = True
        old_input_keys = list(self.last_inputs.keys())
        for key, val, ops, data_type in formatted_inputs:
            # set data
            self.upload(self.input_data_name_of(key), val)
            # compare
            if key in old_input_keys:
                old_input_keys.remove(key)
                if ops.shape(self.last_inputs[key][0]) != ops.shape(val):
                    input_keep_same = False
                    if show_code:
                        print(f'The current "{key}" input shape {ops.shape(val)} is different '
                              f'from the last input shape {ops.shape(self.last_inputs[key][0])}.')
                if self.last_inputs[key][1] != ops:
                    input_keep_same = False
                    if show_code:
                        print(f'The current "{key}" input operation "{ops}" is different '
                              f'from the last operation "{self.last_inputs[key][1]}". ')
            else:
                input_keep_same = False
                if show_code:
                    print(f'The input to a new key "{key}" in {self.host}.')
            new_inputs[key] = (val, ops, data_type)
        self.last_inputs = new_inputs
        if len(old_input_keys):
            input_keep_same = False
            if show_code:
                print(f'The inputs of {old_input_keys} in {self.host} are not provided.')

        return input_keep_same

    def _format_inputs_func(self, formatted_inputs, show_code):
        # codes
        input_func_name = 'input_step'
        host_name = self.host.name
        code_scope = {host_name: self.host}
        code_lines = [f'def {input_func_name}(_i):']
        for key, val, ops, data_type in formatted_inputs:
            if ops == '=':
                line = f'  {host_name}.{key} = {host_name}.{self.input_data_name_of(key)}'
            else:
                line = f'  {host_name}.{key} {ops}= {host_name}.{self.input_data_name_of(key)}'
            if data_type == 'iter':
                line = line + '[_i]'
            code_lines.append(line)

        # function
        code = '\n'.join(code_lines)
        if show_code:
            print(code)
            print(code_scope)
            print()
        exec(compile(code, '', 'exec'), code_scope)
        self.upload(input_func_name, code_scope[input_func_name])
        # results
        self.formatted_funcs['input'] = {
            'func': code_scope[input_func_name],
            'scope': {host_name: self.host},
            'call': [f'{host_name}.{input_func_name}(_i)'],
        }

    def get_input_func(self, formatted_inputs, show_code=False):
        input_keep_same = self._check_inputs_change(formatted_inputs=formatted_inputs, show_code=show_code)
        if not input_keep_same and len(formatted_inputs) > 0:
            self._format_inputs_func(formatted_inputs=formatted_inputs, show_code=show_code)
            need_rebuild = True
        else:
            need_rebuild = False
        return need_rebuild

    def get_monitor_func(self, mon_length, show_code=False):
        mon = self.host.mon
        if mon.num_item > 0:
            monitor_func_name = 'monitor_step'
            host = self.host.name
            code_scope = {host: self.host}
            code_lines = [f'def {monitor_func_name}(_i):']
            for key in mon.item_names:
                if not hasattr(self.host, key):
                    raise errors.ModelUseError(f'{self.host} do not have {key}, '
                                               f'thus it cannot be monitored.')

                # initialize monitor array #
                shape = ops.shape(getattr(self.host, key))
                setattr(mon, key, ops.zeros((mon_length,) + shape))

                # add line #
                line = f'  {host}.mon.{key}[_i] = {host}.{key}'
                code_lines.append(line)

            # function
            code = '\n'.join(code_lines)
            if show_code:
                print(code)
                print(code_scope)
                print()
            exec(compile(code, '', 'exec'), code_scope)
            self.upload(monitor_func_name, code_scope[monitor_func_name])
            # results
            self.formatted_funcs['monitor'] = {
                'func': code_scope[monitor_func_name],
                'scope': {host: self.host},
                'call': [f'{host}.{monitor_func_name}(_i)'],
            }

    def get_steps_func(self, show_code=False):
        for func_name, step in self.steps.items():
            class_args, arguments = utils.get_args(step)
            host_name = self.host.name

            calls = []
            for arg in arguments:
                if hasattr(self.host, arg):
                    calls.append(f'{host_name}.{arg}')
                elif arg in backend.SYSTEM_KEYWORDS:
                    calls.append(arg)
                else:
                    raise errors.ModelDefError(f'Step function "{func_name}" of {self.host} '
                                               f'define an unknown argument "{arg}" which is not '
                                               f'an attribute of {self.host} nor the system keywords '
                                               f'{backend.SYSTEM_KEYWORDS}.')
            self.formatted_funcs[func_name] = {
                'func': step,
                'scope': {host_name: self.host},
                'call': [f'{host_name}.{func_name}({", ".join(calls)})']
            }

    def build(self, formatted_inputs, mon_length, return_code=True, show_code=False):
        # inputs check
        # --
        assert isinstance(formatted_inputs, (tuple, list))
        need_rebuild = self.get_input_func(formatted_inputs, show_code=show_code)
        self.formatted_funcs['need_rebuild'] = need_rebuild

        # the run function does not build before
        # ---
        if self.run_func is None:
            # monitors
            self.get_monitor_func(mon_length, show_code=show_code)

            # steps
            self.get_steps_func(show_code=show_code)

        # reshape the monitor
        self.host.mon.reshape(run_length=mon_length)

        # build the model
        if need_rebuild or self.run_func is None:
            code_scope = dict()
            code_lines = ['def run_func(_t, _i, _dt):']
            for process in self.get_schedule():
                if (process not in self.formatted_funcs) and (process in ['input', 'monitor']):
                    continue
                process_result = self.formatted_funcs[process]
                code_scope.update(process_result['scope'])
                code_lines.extend(process_result['call'])

            # function
            code = '\n  '.join(code_lines)
            if show_code:
                print(code)
                print(code_scope)
                print()
            exec(compile(code, '', 'exec'), code_scope)
            self.run_func = code_scope['run_func']

        if return_code:
            return self.run_func, self.formatted_funcs
        else:
            return self.run_func

    @staticmethod
    def input_data_name_of(key):
        return f'_input_data_of_{key.replace(".", "_")}'


class GeneralNetDriver(drivers.BaseNetDriver):
    """General BrainPy Network Running Driver for NumPy, PyTorch, TensorFlow, etc.
    """

    def __init__(self, host):
        super(GeneralNetDriver, self).__init__(host=host)
        assert hasattr(self.host, 'all_nodes') and isinstance(self.host.all_nodes, dict)
        self.run_func = None

    def build(self, run_length, formatted_inputs, return_code=False, show_code=False):
        """Build the network.

        Parameters
        ----------
        run_length : int
            The running length.
        formatted_inputs : dict
            The user-defined inputs.
        show_code : bool
            Show the formatted code.
        return_code : bool
            Return the code lines and code scope.

        Returns
        -------
        step_func : callable
            The step function.
        """
        if not isinstance(run_length, int):
            raise errors.ModelUseError(f'The running length must be an int, '
                                       f'but we get {type(run_length)}')

        # codes for step function
        need_rebuild = False
        code_scope = {}
        code_lines = ['def run_func(_t, _i, _dt):']
        for obj in self.host.all_nodes.values():
            f, format_funcs = obj.build(inputs=formatted_inputs.get(obj.name, []),
                                        inputs_is_formatted=True,
                                        mon_length=run_length,
                                        return_code=True,
                                        show_code=show_code)
            need_rebuild *= format_funcs['need_rebuild']
            for p in obj.get_schedule():
                if (p not in format_funcs) and (p in ['input', 'monitor']):
                    continue
                p_codes = format_funcs[p]
                code_scope.update(p_codes['scope'])
                code_lines.extend(p_codes['call'])

        # compile the step function
        if (self.run_func is None) or need_rebuild:
            code = '\n  '.join(code_lines)
            if show_code:
                print(code)
                print(code_scope)
                print()
            exec(compile(code, '', 'exec'), code_scope)
            self.run_func = code_scope['run_func']

        if return_code:
            return self.run_func, code_lines, code_scope
        else:
            return self.run_func
