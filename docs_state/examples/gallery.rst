Examples Gallery
================

Welcome to the ``brainpy.state`` examples gallery! Here you'll find complete, runnable examples demonstrating various aspects of computational neuroscience modeling.

All examples are available in the `examples_state/ <https://github.com/brainpy/BrainPy/tree/master/examples_state>`_ directory of the BrainPy repository.

Classical Network Models
-------------------------

These examples reproduce influential models from the computational neuroscience literature.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: E-I Balanced Networks 
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/102_EI_net_1996.py

      Implements the classic excitatory-inhibitory balanced network showing chaotic dynamics.  

      - 80% excitatory, 20% inhibitory neurons
      - Random sparse connectivity
      - Balanced excitation and inhibition
      - Asynchronous irregular firing

      
   .. grid-item-card:: COBA Network (2005)
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/103_COBA_2005.py
      
      Conductance-based synaptic integration in balanced networks.

      - Conductance-based synapses (COBA)
      - Reversal potentials
      - More biologically realistic
      - Stable asynchronous activity


   .. grid-item-card:: CUBA Network (2005)
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/104_CUBA_2005.py

      Current-based synaptic integration (simpler, faster variant).

      - Current-based synapses (CUBA)
      - Faster computation
      - Widely used for large-scale simulations



   .. grid-item-card:: COBA with Hodgkin-Huxley Neurons (2007)
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/106_COBA_HH_2007.py

      More detailed neuron model with sodium and potassium channels.

      - Hodgkin-Huxley neuron dynamics
      - Action potential generation
      - Biophysically detailed
      - Computationally intensive


Oscillations and Rhythms
-------------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Gamma Oscillation (1996)
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/107_gamma_oscillation_1996.py

      Interneuron network generating gamma oscillations (30-80 Hz).

      - Interneuron-based gamma
      - Inhibition-based synchrony
      - Physiologically relevant frequency
      - Network oscillations

   .. grid-item-card:: Synfire Chains (199x)
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/108_synfire_chains_199.py

      Demonstrates reliable spike sequence propagation.

      - Feedforward architecture
      - Reliable spike timing
      - Wave propagation
      - Temporal coding

   .. grid-item-card:: Fast Global Oscillation
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/109_fast_global_oscillation.py

      High-frequency oscillations (>100 Hz) in inhibitory networks.

      - Very fast oscillations
      - Gap junction coupling
      - Inhibitory synchrony
      - Pathological rhythms


Gamma Oscillation Mechanisms (Susin & Destexhe 2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Series of models exploring different gamma generation mechanisms:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Asynchronous Irregular (AI)
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/110_Susin_Destexhe_2021_gamma_oscillation_AI.py

      AI state: No oscillations, irregular firing

      - Background activity state
      - Asynchronous firing
      - No clear rhythm


   .. grid-item-card:: CHING Mechanism
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/111_Susin_Destexhe_2021_gamma_oscillation_CHING.py

      Coherent High-frequency INhibition-based Gamma

      - Coherent inhibition
      - High-frequency gamma
      - Interneuron synchrony


   .. grid-item-card:: ING Mechanism
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/112_Susin_Destexhe_2021_gamma_oscillation_ING.py

      Inhibition-based Gamma

      - Pure inhibitory network
      - Gamma through inhibition
      - Fast synaptic kinetics


   .. grid-item-card:: PING Mechanism
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/113_Susin_Destexhe_2021_gamma_oscillation_PING.py

      Pyramidal-Interneuron Gamma

      - E-I loop generates gamma
      - Most common mechanism
      - Excitatory-inhibitory interaction


**Combined**: `Susin_Destexhe_2021_gamma_oscillation.py <https://github.com/brainpy/BrainPy/tree/master/examples_state/Susin_Destexhe_2021_gamma_oscillation.py>`_ - All mechanisms

**Key Concepts**: Gamma mechanisms, network states, oscillation generation

Spiking Neural Network Training
--------------------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Supervised Learning with Surrogate Gradients
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/200_surrogate_grad_lif.py

      Trains a simple spiking network using surrogate gradients.

      - Surrogate gradient method
      - LIF neuron training
      - Simple classification task
      - Gradient-based learning


   .. grid-item-card:: Fashion-MNIST Classification
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/201_surrogate_grad_lif_fashion_mnist.py

      Trains a spiking network on Fashion-MNIST dataset.

      - Fashion-MNIST dataset
      - Multi-layer SNN
      - Spike-based processing
      - Real-world classification


   .. grid-item-card:: MNIST with Readout Layer
      :link: https://github.com/brainpy/BrainPy/tree/master/examples_state/202_mnist_lif_readout.py

      Uses readout layer for classification.

      - MNIST handwritten digits
      - Specialized readout layer
      - Spike counting
      - Classification from spike rates
