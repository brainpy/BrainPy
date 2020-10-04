# -*- coding: utf-8 -*-


__all__ = ['Connector']


class Connector(object):
    """Abstract connector class."""

    def __call__(self, geom_pre, geom_post):
        raise NotImplementedError

