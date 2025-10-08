# BrainPy Version 3 Examples

This directory contains example scripts demonstrating the capabilities of BrainPy 3.x for simulating and training spiking neural networks.

## Overview

BrainPy 3.x is rewritten based on [brainstate](https://github.com/chaobrain/brainstate) and provides a powerful framework for computational neuroscience and brain-inspired computation.

## Example Categories

### Network Simulations (100-series)

Classic network models demonstrating recurrent dynamics and emergent behaviors:

- **102_EI_net_1996.py** - E/I balanced network from Brunel (1996) and Van Vreeswijk & Sompolinsky (1996)
- **103_COBA_2005.py** - Conductance-based E/I network (COBA model)
- **104_CUBA_2005.py** - Current-based E/I network (CUBA model)
- **106_COBA_HH_2007.py** - COBA network with Hodgkin-Huxley neurons
- **107_gamma_oscillation_1996.py** - Gamma oscillation generation
- **108_synfire_chains_199.py** - Synfire chain propagation
- **109_fast_global_oscillation.py** - Fast global oscillation dynamics

### Gamma Oscillation Models (110-series)

Implementations from Susin & Destexhe (2021) demonstrating different gamma oscillation mechanisms:

- **110_Susin_Destexhe_2021_gamma_oscillation_AI.py** - Asynchronous-Irregular (AI) regime
- **111_Susin_Destexhe_2021_gamma_oscillation_CHING.py** - CHING mechanism
- **112_Susin_Destexhe_2021_gamma_oscillation_ING.py** - Interneuron Gamma (ING)
- **113_Susin_Destexhe_2021_gamma_oscillation_PING.py** - Pyramidal-Interneuron Gamma (PING)

### Spiking Neural Network Training (200-series)

Examples demonstrating training of SNNs using surrogate gradient methods:

- **200_surrogate_grad_lif.py** - Basic surrogate gradient learning with LIF neurons
- **201_surrogate_grad_lif_fashion_mnist.py** - Fashion-MNIST classification with surrogate gradients
- **202_mnist_lif_readout.py** - MNIST classification with LIF network and readout layer

## Requirements

```bash
pip install -U brainpy[cpu]  # or brainpy[cuda12] for GPU support
```

## Usage

Run any example directly:

```bash
python 102_EI_net_1996.py
```

## Key Features Demonstrated

- Building recurrent spiking neural networks
- Neuron models (LIF, Hodgkin-Huxley)
- Synaptic models (exponential, conductance-based, current-based)
- Network projection and connectivity
- Surrogate gradient learning for SNNs
- State management and initialization
- Visualization of network activity

## References

These examples are based on influential papers in computational neuroscience. See individual script headers for specific citations.

## Documentation

For more information, visit the [BrainPy documentation](https://brainpy.readthedocs.io).
