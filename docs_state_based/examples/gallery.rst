Examples Gallery
================

Welcome to the BrainPy 3.0 examples gallery! Here you'll find complete, runnable examples demonstrating various aspects of computational neuroscience modeling.

All examples are available in the `examples_version3/ <https://github.com/brainpy/BrainPy/tree/master/examples_version3>`_ directory of the BrainPy repository.

Classical Network Models
-------------------------

These examples reproduce influential models from the computational neuroscience literature.

E-I Balanced Networks
~~~~~~~~~~~~~~~~~~~~~

**102_EI_net_1996.py** - Van Vreeswijk & Sompolinsky (1996)

Implements the classic excitatory-inhibitory balanced network showing chaotic dynamics.

.. code-block:: python

    # Key features:
    - 80% excitatory, 20% inhibitory neurons
    - Random sparse connectivity
    - Balanced excitation and inhibition
    - Asynchronous irregular firing

:download:`Download <../../examples_version3/102_EI_net_1996.py>`

**Key Concepts**: E-I balance, network dynamics, sparse connectivity

---

COBA Network (2005)
~~~~~~~~~~~~~~~~~~~

**103_COBA_2005.py** - Vogels & Abbott (2005)

Conductance-based synaptic integration in balanced networks.

.. code-block:: python

    # Key features:
    - Conductance-based synapses (COBA)
    - Reversal potentials
    - More biologically realistic
    - Stable asynchronous activity

:download:`Download <../../examples_version3/103_COBA_2005.py>`

**Key Concepts**: COBA synapses, conductance-based models, reversal potentials

---

CUBA Network (2005)
~~~~~~~~~~~~~~~~~~~

**104_CUBA_2005.py** - Vogels & Abbott (2005)

Current-based synaptic integration (simpler, faster variant).

.. code-block:: python

    # Key features:
    - Current-based synapses (CUBA)
    - Faster computation
    - Widely used for large-scale simulations

:download:`Download <../../examples_version3/104_CUBA_2005.py>`

**Alternative**: `104_CUBA_2005_version2.py <../../examples_version3/104_CUBA_2005_version2.py>`_ - Different parameterization

**Key Concepts**: CUBA synapses, current-based models

---

COBA with Hodgkin-Huxley Neurons (2007)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**106_COBA_HH_2007.py** - Conductance-based network with HH neurons

More detailed neuron model with sodium and potassium channels.

.. code-block:: python

    # Key features:
    - Hodgkin-Huxley neuron dynamics
    - Action potential generation
    - Biophysically detailed
    - Computationally intensive

:download:`Download <../../examples_version3/106_COBA_HH_2007.py>`

**Key Concepts**: Hodgkin-Huxley model, ion channels, biophysical detail

Oscillations and Rhythms
-------------------------

Gamma Oscillation (1996)
~~~~~~~~~~~~~~~~~~~~~~~~~

**107_gamma_oscillation_1996.py** - Gamma rhythm generation

Interneuron network generating gamma oscillations (30-80 Hz).

.. code-block:: python

    # Key features:
    - Interneuron-based gamma
    - Inhibition-based synchrony
    - Physiologically relevant frequency
    - Network oscillations

:download:`Download <../../examples_version3/107_gamma_oscillation_1996.py>`

**Key Concepts**: Gamma oscillations, network synchrony, inhibitory networks

---

Synfire Chains (199x)
~~~~~~~~~~~~~~~~~~~~~

**108_synfire_chains_199.py** - Feedforward activity propagation

Demonstrates reliable spike sequence propagation.

.. code-block:: python

    # Key features:
    - Feedforward architecture
    - Reliable spike timing
    - Wave propagation
    - Temporal coding

:download:`Download <../../examples_version3/108_synfire_chains_199.py>`

**Key Concepts**: Synfire chains, feedforward networks, spike timing

---

Fast Global Oscillation
~~~~~~~~~~~~~~~~~~~~~~~

**109_fast_global_oscillation.py** - Ultra-fast network rhythms

High-frequency oscillations (>100 Hz) in inhibitory networks.

.. code-block:: python

    # Key features:
    - Very fast oscillations (>100 Hz)
    - Gap junction coupling
    - Inhibitory synchrony
    - Pathological rhythms

:download:`Download <../../examples_version3/109_fast_global_oscillation.py>`

**Key Concepts**: Fast oscillations, gap junctions, pathological rhythms

Gamma Oscillation Mechanisms (Susin & Destexhe 2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Series of models exploring different gamma generation mechanisms:

**110_Susin_Destexhe_2021_gamma_oscillation_AI.py** - Asynchronous Irregular

.. code-block:: python

    # AI state: No oscillations, irregular firing
    - Background activity state
    - Asynchronous firing
    - No clear rhythm

:download:`Download <../../examples_version3/110_Susin_Destexhe_2021_gamma_oscillation_AI.py>`

---

**111_Susin_Destexhe_2021_gamma_oscillation_CHING.py** - Coherent High-frequency INhibition-based Gamma

.. code-block:: python

    # CHING mechanism
    - Coherent inhibition
    - High-frequency gamma
    - Interneuron synchrony

:download:`Download <../../examples_version3/111_Susin_Destexhe_2021_gamma_oscillation_CHING.py>`

---

**112_Susin_Destexhe_2021_gamma_oscillation_ING.py** - Inhibition-based Gamma

.. code-block:: python

    # ING mechanism
    - Pure inhibitory network
    - Gamma through inhibition
    - Fast synaptic kinetics

:download:`Download <../../examples_version3/112_Susin_Destexhe_2021_gamma_oscillation_ING.py>`

---

**113_Susin_Destexhe_2021_gamma_oscillation_PING.py** - Pyramidal-Interneuron Gamma

.. code-block:: python

    # PING mechanism
    - E-I loop generates gamma
    - Most common mechanism
    - Excitatory-inhibitory interaction

:download:`Download <../../examples_version3/113_Susin_Destexhe_2021_gamma_oscillation_PING.py>`

**Combined**: `Susin_Destexhe_2021_gamma_oscillation.py <../../examples_version3/Susin_Destexhe_2021_gamma_oscillation.py>`_ - All mechanisms

**Key Concepts**: Gamma mechanisms, network states, oscillation generation

Spiking Neural Network Training
--------------------------------

Supervised Learning with Surrogate Gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**200_surrogate_grad_lif.py** - Basic SNN training (SpyTorch tutorial reproduction)

Trains a simple spiking network using surrogate gradients.

.. code-block:: python

    # Key features:
    - Surrogate gradient method
    - LIF neuron training
    - Simple classification task
    - Gradient-based learning

:download:`Download <../../examples_version3/200_surrogate_grad_lif.py>`

**Key Concepts**: Surrogate gradients, SNN training, backpropagation through time

---

Fashion-MNIST Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**201_surrogate_grad_lif_fashion_mnist.py** - Image classification with SNNs

Trains a spiking network on Fashion-MNIST dataset.

.. code-block:: python

    # Key features:
    - Fashion-MNIST dataset
    - Multi-layer SNN
    - Spike-based processing
    - Real-world classification

:download:`Download <../../examples_version3/201_surrogate_grad_lif_fashion_mnist.py>`

**Key Concepts**: Image classification, multi-layer SNNs, practical applications

---

MNIST with Readout Layer
~~~~~~~~~~~~~~~~~~~~~~~~~

**202_mnist_lif_readout.py** - MNIST with specialized readout

Uses readout layer for classification.

.. code-block:: python

    # Key features:
    - MNIST handwritten digits
    - Specialized readout layer
    - Spike counting
    - Classification from spike rates

:download:`Download <../../examples_version3/202_mnist_lif_readout.py>`

**Key Concepts**: Readout layers, spike-based classification, MNIST

Example Categories
------------------

By Difficulty
~~~~~~~~~~~~~

**Beginner** (Start here!)
   - 102_EI_net_1996.py - Simple E-I network
   - 104_CUBA_2005.py - Current-based synapses
   - 200_surrogate_grad_lif.py - Basic training

**Intermediate**
   - 103_COBA_2005.py - Conductance-based synapses
   - 107_gamma_oscillation_1996.py - Network oscillations
   - 201_surrogate_grad_lif_fashion_mnist.py - Image classification

**Advanced**
   - 106_COBA_HH_2007.py - Biophysical detail
   - 113_Susin_Destexhe_2021_gamma_oscillation_PING.py - Complex mechanisms
   - Large-scale simulations (coming soon)

By Topic
~~~~~~~~

**Network Dynamics**
   - E-I balanced networks (102, 103, 104)
   - Oscillations (107, 109, 110-113)
   - Synfire chains (108)

**Synaptic Mechanisms**
   - CUBA models (104)
   - COBA models (103, 106)
   - Different synapse types

**Learning and Training**
   - Surrogate gradients (200, 201, 202)
   - Classification tasks
   - Supervised learning

**Biophysical Models**
   - Hodgkin-Huxley neurons (106)
   - Detailed conductances
   - Realistic parameters

Running Examples
----------------

All examples can be run directly:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/brainpy/BrainPy.git
    cd BrainPy

    # Run an example
    python examples_version3/102_EI_net_1996.py

Or in Jupyter:

.. code-block:: python

    # In Jupyter notebook
    %run examples_version3/102_EI_net_1996.py

Requirements
~~~~~~~~~~~~

Examples require:

- Python 3.10+
- BrainPy 3.0
- matplotlib (for visualization)
- Additional dependencies as noted in examples

Example Structure
-----------------

Most examples follow this structure:

.. code-block:: python

    # 1. Imports
    import brainpy as bp
    import brainstate
    import brainunit as u
    import matplotlib.pyplot as plt

    # 2. Network definition
    class MyNetwork(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            # Define components

        def update(self, input):
            # Define dynamics

    # 3. Setup
    brainstate.environ.set(dt=0.1 * u.ms)
    net = MyNetwork()
    brainstate.nn.init_all_states(net)

    # 4. Simulation
    times = u.math.arange(0*u.ms, 1000*u.ms, dt)
    results = brainstate.transform.for_loop(net.update, times)

    # 5. Visualization
    plt.figure()
    # ... plotting code ...
    plt.show()

Contributing Examples
---------------------

We welcome new examples! To contribute:

1. Fork the BrainPy repository
2. Add your example to ``examples_version3/``
3. Follow naming convention: ``NNN_descriptive_name.py``
4. Include documentation at the top
5. Submit a pull request

Example Template:

.. code-block:: python

    # Copyright 2024 BrainX Ecosystem Limited.
    # Licensed under Apache License 2.0

    """
    Short description of the example.

    This example demonstrates:
    - Feature 1
    - Feature 2

    References:
    - Citation if reproducing paper
    """

    # Your code here...

Additional Resources
--------------------

**Tutorials**
   For step-by-step learning, see :doc:`../tutorials/basic/01-lif-neuron`

**API Documentation**
   For detailed API reference, see :doc:`../api/neurons`

**Core Concepts**
   For architectural understanding, see :doc:`../core-concepts/architecture`

**Migration Guide**
   For updating from 2.x, see :doc:`../migration/migration-guide`

Browse All Examples
-------------------

View all examples on GitHub:

`BrainPy Examples (Version 3.0) <https://github.com/brainpy/BrainPy/tree/master/examples_version3>`_

For more extensive examples and notebooks:

`BrainPy Examples Repository <https://github.com/brainpy/examples>`_

Getting Help
------------

If you have questions about examples:

- Open an issue on GitHub
- Check existing discussions
- Read the tutorials
- Consult the documentation

Happy modeling! 🧠
