Core Concepts
=============

This section provides an in-depth exploration of the fundamental concepts underlying ``brainpy.state``.
Understanding these core principles will help you build more sophisticated and efficient neural network
models.

``brainpy.state`` introduces a State-based programming paradigm that simplifies the development of
spiking neural networks while improving performance and scalability. The concepts covered here form
the foundation for all neural modeling work in this framework.


Overview
--------

The core concepts of ``brainpy.state`` include:

- **Architecture**: The overall structure and design principles of State-based neural networks, including how components interact and compose together
- **Neurons**: Building blocks for neural computation, including different neuron models and their state representations
- **Synapses**: Connections between neurons that transmit signals and implement learning rules
- **Projections**: Network-level structures that organize and manage connections between populations of neurons
- **State Management**: The powerful state handling system that enables efficient simulation and flexible model composition


Why these concepts matter
--------------------------

Mastering these core concepts will enable you to:

- Design complex neural networks with clean, maintainable code
- Leverage the State-based paradigm for improved performance
- Understand how to compose neurons, synapses, and projections effectively
- Manage model state efficiently during simulation
- Seamlessly integrate with the BrainX ecosystem

Each concept builds upon the previous ones, so we recommend reading them in order if you're new to ``brainpy.state``.


.. toctree::
   :hidden:
   :maxdepth: 1

   architecture.ipynb
   neurons.ipynb
   synapses.ipynb
   projections.ipynb
   state-management.ipynb


