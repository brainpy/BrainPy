# -*- coding: utf-8 -*-

import jax
from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.simulation.datastructures.collections import DataCollector
from brainpy.backend.bpjax.core.function import Function


class VMAP(DynamicSystem):
    """Vectorize module takes a function or a module and
    compiles it for running in parallel on a single device.

    Parameters
    ----------

    f : DynamicSystem, function
      The function or the module to compile for vectorization.
    all_vars : DataCollector
      The Collection of variables used by the function or module.
      This argument is required for functions.
    batch_axis : tuple of int, int, tuple of None
      Tuple of int or None for each of f's input arguments:
      the axis to use as batch during vectorization. Use
      None to automatically broadcast.
    """

    def __init__(self, f, all_vars=None, batch_axis=(0,)):
        if not isinstance(f, DynamicSystem):
            if all_vars is None:
                raise ValueError('You must supply the VarCollection used by the function f.')
            f = Function(f, all_vars)

        def vmap(all_data, random_list, *args):
          self.all_vars.assign(all_data)
          self.all_vars.subset(RandomState).assign(random_list)
          return f(*args), self.all_vars.all_data()

        fargs = positional_args_names(f)
        assert len(batch_axis) >= len(fargs), f'The batched argument must be specified for all of {f} arguments {fargs}'
        self.batch_axis = batch_axis
        self.batch_axis_argnums = [(x, v) for x, v in enumerate(batch_axis) if v is not None]
        assert self.batch_axis_argnums, f'No arguments to function {f} are vectorizable'
        self.all_vars = all_vars or f.vars()
        self._call = jax.vmap(vmap, (None, 0) + batch_axis)
        self.__wrapped__ = f

    def __call__(self, *args):
        """Call the vectorized version of the function or module."""
        assert len(args) == len(self.batch_axis), f'Number of arguments passed {len(args)} must match ' \
                                                  f'batched {len(self.batch_axis)}'
        nsplits = args[self.batch_axis_argnums[0][0]].shape[self.batch_axis_argnums[0][1]]
        output, changes = self._call(self.all_vars.all_data(),
                                     [v.split(nsplits) for v in self.all_vars.subset(RandomState)],
                                     *args)
        for v, u in zip(self.all_vars, changes):
            v.reduce(u)
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}(f={self.__wrapped__}, batch_axis={self.batch_axis})'
