# -*- coding: utf-8 -*-

from jax.lib import xla_client


# Register the CPU XLA custom calls
try:
  from brainpy import cpu_ops

  for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")
except ImportError:
  cpu_ops = None

# Register the GPU XLA custom calls
try:
  from brainpy import gpu_ops
except ImportError:
  gpu_ops = None
else:
  for _name, _value in gpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")
