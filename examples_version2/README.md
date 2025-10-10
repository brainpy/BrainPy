# BrainPy Version 2 Examples

This directory contains example scripts demonstrating the capabilities of BrainPy 2.x for brain dynamics programming.

## Overview

These examples showcase BrainPy 2.x functionality including dynamics simulation, analysis, and training. BrainPy 2.x is maintained for backward compatibility, but new projects should consider using BrainPy 3.x.

## Important Note

**As of September 2025, BrainPy has been upgraded to version 3.x.** To use these examples with BrainPy 3.x, update your imports:

```python
import brainpy as bp
import brainpy.math as bm
```

## Example Categories

### Dynamics Simulation

Network simulation examples demonstrating various neural models and dynamics:

- **hh_model.py** - Hodgkin-Huxley neuron model
- **ei_nets.py** - Excitatory-inhibitory networks
- **COBA.py** - Conductance-based network model
- **stdp.py** - Spike-timing-dependent plasticity
- **decision_making_network.py** - Decision-making circuit
- **whole_brain_simulation_with_fhn.py** - Whole-brain simulation with FitzHugh-Nagumo model
- **whole_brain_simulation_with_sl_oscillator.py** - Whole-brain simulation with Stuart-Landau oscillator

### Dynamics Analysis

Phase plane and bifurcation analysis of neural models:

- **1d_qif.py** - 1D Quadratic Integrate-and-Fire model analysis
- **2d_fitzhugh_nagumo_model.py** - 2D FitzHugh-Nagumo phase plane analysis
- **2d_mean_field_QIF.py** - 2D mean-field QIF analysis
- **3d_reduced_trn_model.py** - 3D reduced thalamic reticular nucleus model
- **4d_HH_model.py** - 4D Hodgkin-Huxley model analysis
- **highdim_RNN_Analysis.py** - High-dimensional RNN dynamics analysis

### Dynamics Training

Training examples for recurrent networks and reservoir computing:

- **echo_state_network.py** - Echo State Network (reservoir computing)
- **integrator_rnn.py** - RNN for integration task
- **reservoir-mnist.py** - MNIST classification with reservoir computing
- **Sussillo_Abbott_2009_FORCE_Learning.py** - FORCE learning algorithm
- **Song_2016_EI_RNN.py** - E/I RNN training
- **integrate_brainpy_into_flax-lif.py** - Integration with Flax (LIF neurons)
- **integrate_brainpy_into_flax-convlstm.py** - Integration with Flax (ConvLSTM)
- **integrate_flax_into_brainpy.py** - Using Flax models in BrainPy

### Training ANN Models

Artificial neural network training examples:

- **mnist-cnn.py** - CNN for MNIST classification
- **mnist_ResNet.py** - ResNet for MNIST classification

### Training SNN Models

Spiking neural network training examples:

- **spikebased_bp_for_cifar10.py** - Spike-based backpropagation for CIFAR-10
- **readme.md** - Additional SNN training documentation

## Requirements

```bash
pip install -U brainpy[cpu]  # or brainpy[cuda12] for GPU
```

For version 3.x with 2.x compatibility:

```bash
pip install -U brainpy[cpu]
# Then use: import brainpy as bp
```

## Usage

Run any example directly:

```bash
python dynamics_simulation/hh_model.py
```

Or with version 3.x (examples may need import updates):

```bash
# Modify imports in the script first, then run
python dynamics_simulation/ei_nets.py
```

## Key Concepts Demonstrated

- **Dynamics Simulation**: Simulating neural circuits and network dynamics
- **Dynamics Analysis**: Phase plane analysis, bifurcation analysis, fixed points
- **Reservoir Computing**: Echo State Networks and Liquid State Machines
- **Network Training**: Gradient-based and FORCE learning for RNNs
- **SNN Training**: Surrogate gradient methods for spiking networks
- **Framework Integration**: Using BrainPy with other frameworks (Flax, JAX)

## Documentation

- [BrainPy 2.x Documentation](https://brainpy-v2.readthedocs.io)
- [BrainPy 3.x Documentation](https://brainpy.readthedocs.io)
- [BrainPy Ecosystem](https://brainmodeling.readthedocs.io)

## Migrating to Version 3.x

For new projects, we recommend using BrainPy 3.x which offers improved performance and a cleaner API. See the migration guide in the main documentation.

## Support

For questions and support, please visit the [BrainPy GitHub repository](https://github.com/brainpy/BrainPy).
