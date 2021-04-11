# -*- coding: utf-8 -*-

from brainpy import errors
from brainpy.backend import ops

__all__ = [
    'Monitor'
]


class Monitor(object):
    """The basic Monitor class to store the past variable trajectories.
    """

    def __init__(self, variables):
        item_names = []
        mon_indices = []
        item_content = {}
        if variables is not None:
            if isinstance(variables, (list, tuple)):
                for var in variables:
                    if isinstance(var, str):
                        item_names.append(var)
                        mon_indices.append(None)
                        item_content[var] = ops.zeros((1, 1))
                    elif isinstance(var, (tuple, list)):
                        item_names.append(var[0])
                        mon_indices.append(var[1])
                        item_content[var[0]] = ops.zeros((1, 1))
                    else:
                        raise errors.ModelUseError(f'Unknown monitor item: {str(var)}')
            elif isinstance(variables, dict):
                for k, v in variables.items():
                    item_names.append(k)
                    mon_indices.append(v)
                    item_content[k] = ops.zeros((1, 1))
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
        if key in ['ts', 'item_names', 'item_indices', 'item_contents', 'num_item']:
            object.__setattr__(self, key, value)
        elif key in self.item_contents:
            self.item_contents[key] = value
        else:
            object.__setattr__(self, key, value)

    def reshape(self, run_length):
        for var in self.item_names:
            val = getattr(self, var)
            shape = ops.shape(val)
            if run_length < shape[0]:
                setattr(self, var, val[:run_length])
            elif run_length > shape[0]:
                append = ops.zeros((run_length - shape[0],) + shape[1:])
                setattr(self, var, ops.vstack([val, append]))

#
# class Monitor(tools.DictPlus):
#     """The basic Monitor class to store the past variable trajectories.
#     """
#
#     def __init__(self, variables):
#         mon_items = []
#         mon_indices = []
#         item_content = {}
#         if variables is not None:
#             if isinstance(variables, (list, tuple)):
#                 for var in variables:
#                     if isinstance(var, str):
#                         mon_items.append(var)
#                         mon_indices.append(None)
#                         item_content[var] = ops.zeros((1, 1))
#                     elif isinstance(var, (tuple, list)):
#                         mon_items.append(var[0])
#                         mon_indices.append(var[1])
#                         item_content[var[0]] = ops.zeros((1, 1))
#                     else:
#                         raise errors.ModelUseError(f'Unknown monitor item: {str(var)}')
#             elif isinstance(variables, dict):
#                 for k, v in variables.items():
#                     mon_items.append(k)
#                     mon_indices.append(v)
#                     item_content[k] = ops.zeros((1, 1))
#             else:
#                 raise errors.ModelUseError(f'Unknown monitors type: {type(variables)}')
#         super(Monitor, self).__init__(ts=None,
#                                       vars=mon_items,
#                                       indices=mon_indices,
#                                       num_item=len(item_content),
#                                       **item_content)
#
#     def reshape(self, run_length):
#         for var in self['vars']:
#             val = self[var]
#             shape = ops.shape(val)
#             if run_length < shape[0]:
#                 self[var] = val[:run_length]
#             elif run_length > shape[0]:
#                 append = ops.zeros((run_length - shape[0],) + shape[1:])
#                 self[var] = ops.vstack([val, append])
