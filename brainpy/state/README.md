# `brainpy.state`

## Overview

The `brainpy.state` module provides a state-based programming interface for brain dynamics modeling in BrainPy, built on [`brainstate`](https://github.com/chaobrain/brainstate).

Its motivation is compatibility with the [BrainX ecosystem](https://brainx.chaobrain.com): `brainpy.state` models carry physical units through [`brainunit`](https://github.com/chaobrain/brainunit), compose with `brainstate` transformations, and can be trained online with [`braintrace`](https://github.com/chaobrain/braintrace).

State-based programming offers an alternative paradigm for defining and managing neural models, emphasizing explicit state management and transformations for building complex brain dynamics systems.

`brainpy.state` is the recommended starting point for new `State`-based, differentiable models; the classic `DynamicalSystem`-based `brainpy` API (`brainpy.dyn`, `brainpy.math`, `brainpy.integrators`) is unchanged and fully supported.

## Features

- **Explicit State Management**: Clear separation between model state and computation logic
- **Composable State Transformations**: Build complex models from simple, reusable state components
- **JAX-compatible**: Fully compatible with JAX's functional programming paradigm and JIT compilation
- **Hardware Acceleration**: Leverage CPU, GPU, and TPU acceleration through JAX backend

## Documentation

- **State-based Documentation**: https://brainpy.readthedocs.io/state/index.html
- **API Reference**: https://brainpy.readthedocs.io/apis/state.html
- **BrainPy and BrainX**: https://brainx.chaobrain.com/summ/brainpy-to-brainx.html
- **Migration notes**: https://brainx.chaobrain.com/summ/brainpy-migration-notes.html

## Installation

The `brainpy.state` module is included when you install BrainPy:

```bash
pip install brainpy -U
```

## Usage

```python
import brainpy

neuron = brainpy.state.LIF(100)
```

## Support

- **Bug Reports**: Please report issues at https://github.com/brainpy/BrainPy/issues

## License

Copyright 2025 BrainX Ecosystem Limited. Licensed under the Apache License, Version 2.0.
