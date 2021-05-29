# -*- coding: utf-8 -*-

from brainpy.backend import ops

import tensorflow as tf

__all__ = []

ops.set_buffer('tensorflow',
               clip=tf.clip_by_value,
               unsqueeze=tf.expand_dims,
               squeeze=tf.squeeze
               )
