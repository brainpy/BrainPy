# -*- coding: utf-8 -*-

from brainpy import backend
from brainpy import errors
from brainpy import tools

__all__ = [
    'Monitor'
]


class Monitor(tools.DictPlus):
    """The basic Monitor class to store the past variable trajectories.
    """
    def __init__(self, variables):
        mon_items = []
        mon_indices = []
        item_content = {}
        if variables is not None:
            if isinstance(variables, (list, tuple)):
                for var in variables:
                    if isinstance(var, str):
                        mon_items.append(var)
                        mon_indices.append(None)
                        item_content[var] = backend.zeros((1, 1))
                    elif isinstance(var, (tuple, list)):
                        mon_items.append(var[0])
                        mon_indices.append(var[1])
                        item_content[var[0]] = backend.zeros((1, 1))
                    else:
                        raise errors.ModelUseError(f'Unknown monitor item: {str(var)}')
            elif isinstance(variables, dict):
                for k, v in variables.items():
                    mon_items.append(k)
                    mon_indices.append(v)
                    item_content[k] = backend.zeros((1, 1))
            else:
                raise errors.ModelUseError(f'Unknown monitors type: {type(variables)}')
        super(Monitor, self).__init__(ts=None,
                                      vars=mon_items,
                                      indices=mon_indices,
                                      num_item=len(item_content),
                                      **item_content)

    def reshape(self, run_length):
        for var in self['vars']:
            val = self[var]
            shape = backend.shape(val)
            if run_length < shape[0]:
                self[var] = val[:run_length]
            elif run_length > shape[0]:
                append = backend.zeros((run_length - shape[0],) + shape[1:])
                self[var] = backend.vstack([val, append])
