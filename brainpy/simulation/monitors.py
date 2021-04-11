# -*- coding: utf-8 -*-

from brainpy import errors
from brainpy.backend import ops

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

    """

    def __init__(self, target, variables):
        self.target = target
        for mon_var in variables:
            if not hasattr(target, mon_var):
                raise errors.ModelDefError(f"Item {mon_var} isn't defined in model {target}, "
                                           f"so it can not be monitored.")

        item_names = []
        mon_indices = []
        item_content = {}
        if variables is not None:
            if isinstance(variables, (list, tuple)):
                for mon_var in variables:
                    if isinstance(mon_var, str):
                        var_data = getattr(target, mon_var)
                        mon_key = mon_var
                        mon_idx = None
                        mon_shape = ops.shape(var_data)
                    elif isinstance(mon_var, (tuple, list)):
                        mon_key = mon_var[0]
                        var_data = getattr(target, mon_key)
                        mon_idx = mon_var[1]
                        mon_shape = ops.shape(mon_idx)  # TODO: matrix index
                    else:
                        raise errors.ModelUseError(f'Unknown monitor item: {str(mon_var)}')
                    item_names.append(mon_key)
                    mon_indices.append(mon_idx)
                    dtype = var_data.dtype if hasattr(var_data, 'dtype') else None
                    item_content[mon_var] = ops.zeros((1,) + mon_shape, dtype=dtype)
            elif isinstance(variables, dict):
                for k, v in variables.items():
                    item_names.append(k)
                    mon_indices.append(v)
                    if v is None:
                        shape = ops.shape(getattr(target, k))
                    else:
                        shape = ops.shape(v)
                    val_data = getattr(target, k)
                    dtype = val_data.dtype if hasattr(val_data, 'dtype') else None
                    item_content[k] = ops.zeros((1,) + shape, dtype=dtype)
            else:
                raise errors.ModelUseError(f'Unknown monitors type: {type(variables)}')

        self.ts = None
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
        if key in ['target', 'ts', 'item_names', 'item_indices', 'item_contents', 'num_item']:
            object.__setattr__(self, key, value)
        elif key in self.item_contents:
            self.item_contents[key] = value
        else:
            object.__setattr__(self, key, value)
