# `brainpy.state` - State-based Brain Dynamics Programming

## Overview

The `brainpy.state` module provides a state-based programming interface for brain dynamics modeling in BrainPy. This module is maintained as a separate package [`brainpy_state`](https://github.com/chaobrain/brainpy.state) and re-exported through BrainPy for seamless integration.

State-based programming offers an alternative paradigm for defining and managing neural models, emphasizing explicit state management and transformations for building complex brain dynamics systems.

## Features

- **Explicit State Management**: Clear separation between model state and computation logic
- **Composable State Transformations**: Build complex models from simple, reusable state components
- **JAX-compatible**: Fully compatible with JAX's functional programming paradigm and JIT compilation
- **Hardware Acceleration**: Leverage CPU, GPU, and TPU acceleration through JAX backend

## Documentation

For comprehensive documentation on state-based programming in BrainPy, please visit:

- **State-based Documentation**: https://brainpy-state.readthedocs.io/
- **Main BrainPy Documentation**: https://brainpy.readthedocs.io/

## Source Repository

This module is maintained in a separate repository:

- **GitHub**: https://github.com/chaobrain/brainpy.state

## Installation

The `brainpy.state` module is included when you install BrainPy:

```bash
pip install brainpy -U
```

For development or to install the state module separately:

```bash
pip install brainpy_state -U
```

## Usage

Import the state module from BrainPy:

```python
import brainpy as bp
from brainpy import state

# Use state-based components
# (See documentation for detailed examples)
```

## Support

- **Bug Reports**: Please report issues at https://github.com/brainpy/BrainPy/issues
- **State Module Issues**: For state-specific issues, see https://github.com/chaobrain/brainpy.state/issues

## License

Copyright 2025 BrainX Ecosystem Limited. Licensed under the Apache License, Version 2.0.
