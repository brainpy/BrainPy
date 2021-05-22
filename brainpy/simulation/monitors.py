# -*- coding: utf-8 -*-

from brainpy import errors
from brainpy.backend import ops

from . import utils

__all__ = [
    'Monitor'
]


class Monitor(object):
    """The basic Monitor class to store the past variable trajectories.

    Users can instance a monitor object by multiple ways:

    1. list of strings.

    >>> Monitor(target=..., variables=['a', 'b', 'c'])

    2. list of strings and string + indices

    >>> Monitor(target=..., variables=['a', ('b', ops.as_tensor([1,2,3])), 'c'])

    3. a dictionary with the format of {key: indices}

    >>> Monitor(target=..., variables={'a': None, 'b': ops.as_tensor([1,2,3])})

    :py:class:`brainpy.simulation.Monitor` records any target variable with an
    array/tensore with the shape of `(num_time_step, variable_size)`. This means
    any variable, no matter what's the shape of the data (vector, matrix,
    3D tensores), will be reshaped into a one-dimensional vector.

    """

    _KEYWORDS = ['target', '_variables', 'ts', 'num_item', '_KEYWORDS',
                 'item_names', 'item_indices', 'item_contents', ]

    def __init__(self, target, variables):
        self.target = target
        self._variables = variables

        self.ts = None
        self.item_names = None
        self.item_indices = None
        self.item_contents = None
        self.num_item = len(variables)

    def check(self, mon_key):
        # check
        if mon_key in self._KEYWORDS:
            raise ValueError(f'"{mon_key}" is a keyword in Monitor class. '
                             f'Please change to another name.')
        if not hasattr(self.target, mon_key):
            raise errors.ModelDefError(f"Item \"{mon_key}\" isn't defined in model "
                                       f"{self.target}, so it can not be monitored.")

    def build(self):
        item_names = []
        mon_indices = []
        item_content = {}
        if self._variables is not None:
            if isinstance(self._variables, (list, tuple)):
                for mon_var in self._variables:
                    # users monitor a variable by a string
                    if isinstance(mon_var, str):
                        var_data = getattr(self.target, mon_var)
                        mon_key = mon_var
                        mon_idx = None
                        mon_shape = (utils.size2len(ops.shape(var_data)), )
                    # users monitor a variable by a tuple: `('b', ops.as_tensor([1,2,3]))`
                    elif isinstance(mon_var, (tuple, list)):
                        mon_key = mon_var[0]
                        var_data = getattr(self.target, mon_key)
                        mon_idx = mon_var[1]
                        if not isinstance(mon_idx, int) or len(ops.shape(mon_idx)) != 1:
                            raise errors.ModelUseError(f'Monitor item index only supports an int or '
                                                       f'a one-dimensional vector, not {str(mon_var)}')
                        mon_shape = ops.shape(mon_idx)
                    else:
                        raise errors.ModelUseError(f'Unknown monitor item: {str(mon_var)}')

                    self.check(mon_key)
                    item_names.append(mon_key)
                    mon_indices.append(mon_idx)
                    dtype = var_data.dtype if hasattr(var_data, 'dtype') else None
                    item_content[mon_var] = ops.zeros((1,) + mon_shape, dtype=dtype)
            elif isinstance(self._variables, dict):
                # users monitor a variable by a dict: `{'a': None, 'b': ops.as_tensor([1,2,3])}`
                for mon_key, mon_idx in self._variables.items():
                    item_names.append(mon_key)
                    mon_indices.append(mon_idx)
                    if mon_idx is None:
                        shape = ops.shape(getattr(self.target, mon_key))
                    else:
                        shape = ops.shape(mon_idx)
                    shape = (utils.size2len(shape), )
                    val_data = getattr(self.target, mon_key)
                    dtype = val_data.dtype if hasattr(val_data, 'dtype') else None
                    item_content[mon_key] = ops.zeros((1,) + shape, dtype=dtype)
            else:
                raise errors.ModelUseError(f'Unknown monitors type: {type(self._variables)}')

        self.item_names = item_names
        self.item_indices = mon_indices
        self.item_contents = item_content
        self.num_item = len(item_content)

    def __getattr__(self, item):
        item_contents = super(Monitor, self).__getattribute__('item_contents')
        if item in item_contents:
            return item_contents[item]
        else:
            super(Monitor, self).__getattribute__(item)

    def __setattr__(self, key, value):
        item_contents = super(Monitor, self).__getattribute__('item_contents')
        if key in item_contents:
            item_contents[key] = value
        else:
            object.__setattr__(self, key, value)
