# BrainPy Version 3 Documentation

This directory contains documentation for BrainPy 3.x, the latest major version of BrainPy.

## Overview

BrainPy 3.x is a flexible, efficient, and extensible framework for computational neuroscience and brain-inspired computation. It has been completely rewritten based on [brainstate](https://github.com/chaobrain/brainstate) (since August 2025) and provides powerful capabilities for building, simulating, and training spiking neural networks.

## Documentation Contents

This directory contains tutorial notebooks and API documentation:

### Tutorials (Bilingual: English & Chinese)

#### SNN Simulation
- **snn_simulation-en.ipynb** - Building and simulating spiking neural networks (English)
- **snn_simulation-zh.ipynb** - 构建和模拟脉冲神经网络 (Chinese)

#### SNN Training
- **snn_training-en.ipynb** - Training spiking neural networks with surrogate gradients (English)
- **snn_training-zh.ipynb** - 使用代理梯度训练脉冲神经网络 (Chinese)

#### Checkpointing
- **checkpointing-en.ipynb** - Saving and loading model states (English)
- **checkpointing-zh.ipynb** - 保存和加载模型状态 (Chinese)

### API Reference
- **apis.rst** - Complete API documentation
- **index.rst** - Main documentation entry point

## Key Features in Version 3.x

- Built on [brainstate](https://github.com/chaobrain/brainstate) for improved state management
- Enhanced support for spiking neural networks
- Streamlined API for building neural models
- Improved performance and scalability
- Better integration with JAX ecosystem
- Support for GPU/TPU acceleration

## Installation

```bash
# CPU version
pip install -U brainpy[cpu]

# GPU version (CUDA 12)
pip install -U brainpy[cuda12]

# GPU version (CUDA 13)
pip install -U brainpy[cuda13]

# TPU version
pip install -U brainpy[tpu]

# Full ecosystem
pip install -U BrainX
```

## Quick Start

```python
import brainpy
import brainstate
import brainunit as u

# Define a simple LIF neuron
neuron = brainpy.LIF(100, V_rest=-60.*u.mV, V_th=-50.*u.mV)

# Initialize and simulate
brainstate.nn.init_all_states(neuron)
spikes = neuron(1.*u.mA)
```

## Migration from Version 2.x

If you're migrating from BrainPy 2.x, the API has changed significantly. See the migration guide in the main documentation for details.

To use legacy 2.x APIs in version 3.x:

```python
import brainpy as bp
import brainpy.math as bm
```

## Running Notebooks

The tutorial notebooks can be run using Jupyter:

```bash
jupyter notebook snn_simulation-en.ipynb
```

Or with JupyterLab:

```bash
jupyter lab
```

## Building Documentation

If you need to build the documentation locally, this directory is part of the larger documentation system. Please refer to the main documentation build instructions.

## Learn More

- [Main Documentation](https://brainpy.readthedocs.io)
- [BrainPy GitHub](https://github.com/brainpy/BrainPy)
- [BrainState GitHub](https://github.com/chaobrain/brainstate)
- [BrainPy Ecosystem](https://brainmodeling.readthedocs.io)

## Contributing

Contributions to documentation are welcome! Please submit issues or pull requests to the [BrainPy repository](https://github.com/brainpy/BrainPy).
