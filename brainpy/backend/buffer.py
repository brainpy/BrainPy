# -*- coding: utf-8 -*-


__all__ = [
    'set_buffer',
    'get_buffer',
]


BUFFER = {}


def set_buffer(backend, ops):
    global BUFFER
    assert isinstance(ops, dict), '"ops" must be dictionary with the format of (key, func).'
    for key, val in ops.items():
        assert callable(val), '"ops" must be dictionary with the format of (key, func).'
    if backend in BUFFER:
        BUFFER[backend].update(ops)
    else:
        BUFFER[backend] = {key: val for key, val in ops.items()}


def get_buffer(backend):
    return BUFFER.get(backend, dict())


