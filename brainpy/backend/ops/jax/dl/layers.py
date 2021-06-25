# -*- coding: utf-8 -*-


from brainpy.simulation.brainobjects.container import Container


class Sequential(Container):
    """Executes modules in the order they were passed to the constructor."""

    @staticmethod
    def run_layer(layer: int, f: Callable, args: List, kwargs: Dict):
        try:
            return f(*args, **util.local_kwargs(kwargs, f))
        except Exception as e:
            raise type(e)(f'Sequential layer[{layer}] {f} {e}') from e

    def update(self, *args, **kwargs) -> Union[JaxArray, List[JaxArray]]:
        """Execute the sequence of operations contained on ``*args`` and ``**kwargs`` and return result."""
        if not self:
            return args if len(args) > 1 else args[0]
        for i, f in enumerate(self[:-1]):
            args = self.run_layer(i, f, args, kwargs)
            if not isinstance(args, tuple):
                args = (args,)
        return self.run_layer(len(self) - 1, self[-1], args, kwargs)

    def __getitem__(self, key: Union[int, slice]):
        value = list.__getitem__(self, key)
        if isinstance(key, slice):
            return Sequential(value)
        return value
