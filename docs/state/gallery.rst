BrainPy-style Examples
======================

Complete, runnable examples built with the native ``brainpy.state`` API:
composable neurons, synapses, AlignPre/AlignPost projections, and
surrogate-gradient training. Every script lives in the `examples/state/
<https://github.com/brainpy/BrainPy/tree/main/examples/state>`_
directory of the repository.

Each card is tagged along two axes:

.. admonition:: How to read the tags
   :class: note

   - **World** — **Simulation** reproduces neuroscience dynamics; **Training**
     learns weights by gradient descent through the spiking model.
   - **Difficulty** — *beginner* (single mechanism, short), *intermediate*
     (a full network or training loop), *advanced* (multi-mechanism models or
     paper-scale reproductions).

   These tags describe scope, not stability — every example is production-ready.

If you are new here, start with the :doc:`/state/tutorials/01-first-neuron`
tutorial track, then come back to
adapt one of these.

Balanced & E/I networks
-----------------------

The canonical excitatory-inhibitory balanced networks — the natural home of the
AlignPost projection (see :doc:`/state/concepts/alignpre-alignpost`).

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: COBA balanced network (2005)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/103_COBA_2005.py

      Vogels-Abbott conductance-based E/I network with ``AlignPostProj`` +
      ``Expon`` + ``COBA``. The reference simulation pattern.

      **Simulation** · beginner

   .. grid-item-card:: CUBA balanced network (2005)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/104_CUBA_2005.py

      The current-based variant of Vogels-Abbott — simpler and faster than COBA.

      **Simulation** · beginner

   .. grid-item-card:: CUBA balanced network (variant)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/104_CUBA_2005_version2.py

      A second construction of the CUBA network showing an alternative wiring
      style.

      **Simulation** · intermediate

   .. grid-item-card:: EI balanced network (1996)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/102_EI_net_1996.py

      Van Vreeswijk & Sompolinsky balanced network exhibiting chaotic
      asynchronous-irregular firing.

      **Simulation** · intermediate

   .. grid-item-card:: COBA with Hodgkin-Huxley neurons (2007)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/106_COBA_HH_2007.py

      The COBA benchmark rebuilt on biophysically detailed Hodgkin-Huxley cells.

      **Simulation** · intermediate

Oscillations & rhythms
----------------------

Network rhythms — gamma generation, synchrony, and fast oscillations.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Interneuron gamma (1996)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/107_gamma_oscillation_1996.py

      Wang & Buzsaki inhibition-based gamma (30-80 Hz) in a hippocampal
      interneuron network.

      **Simulation** · intermediate

   .. grid-item-card:: Synfire chains (1999)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/108_synfire_chains_199.py

      Diesmann et al. reliable propagation of synchronous spike volleys through a
      feedforward chain.

      **Simulation** · intermediate

   .. grid-item-card:: Fast global oscillation
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/109_fast_global_oscillation.py

      Brunel & Hakim fast (>100 Hz) global oscillations in sparsely-connected
      inhibitory networks with low firing rates.

      **Simulation** · intermediate

   .. grid-item-card:: Gamma mechanisms — AI state (2021)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/110_Susin_Destexhe_2021_gamma_oscillation_AI.py

      Susin & Destexhe asynchronous-irregular background state (no rhythm).

      **Simulation** · advanced

   .. grid-item-card:: Gamma mechanisms — CHING (2021)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/111_Susin_Destexhe_2021_gamma_oscillation_CHING.py

      Susin & Destexhe coherent high-frequency inhibition-based gamma.

      **Simulation** · advanced

   .. grid-item-card:: Gamma mechanisms — ING (2021)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/112_Susin_Destexhe_2021_gamma_oscillation_ING.py

      Susin & Destexhe pure-inhibitory (ING) gamma generation.

      **Simulation** · advanced

   .. grid-item-card:: Gamma mechanisms — PING (2021)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/113_Susin_Destexhe_2021_gamma_oscillation_PING.py

      Susin & Destexhe pyramidal-interneuron (PING) gamma — the E-I loop.

      **Simulation** · advanced

   .. grid-item-card:: Gamma mechanisms — combined driver (2021)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/Susin_Destexhe_2021_gamma_oscillation.py

      The combined Susin & Destexhe model exploring all four gamma regimes in one
      script.

      **Simulation** · advanced

Reproductions & large models
----------------------------

Paper-scale simulations and signal-propagation studies.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Cortical microcircuit model
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/203_cortical_model.py

      A multi-layer cortical column model with layer-specific LIF populations and
      connectivity.

      **Simulation** · advanced

   .. grid-item-card:: Inter-areal propagation (Joglekar 2018)
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/204_joglekar_2018_propagation.py

      Joglekar et al. graded vs all-or-none signal propagation across a
      connectome-constrained network of cortical areas.

      **Simulation** · advanced

Training spiking networks
-------------------------

Gradient-based training through the spiking model with surrogate gradients (see
:doc:`/state/concepts/differentiability`).

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Surrogate-gradient LIF
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/200_surrogate_grad_lif.py

      Train a small spiking network on a synthetic task with surrogate gradients
      (the spytorch tutorial 1 reproduction).

      **Training** · beginner

   .. grid-item-card:: Fashion-MNIST SNN
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/201_surrogate_grad_lif_fashion_mnist.py

      A multi-layer spiking network classifying Fashion-MNIST end to end
      (spytorch tutorials 2 & 3).

      **Training** · intermediate

   .. grid-item-card:: MNIST with a rate readout
      :link: https://github.com/brainpy/BrainPy/blob/main/examples/state/202_mnist_lif_readout.py

      MNIST classification with a ``LeakyRateReadout`` decoding spikes into class
      scores.

      **Training** · intermediate

See Also
--------

- :doc:`/state/index` — tutorials and task-oriented how-to guides.
- :doc:`/state/concepts/alignpre-alignpost` — the projection design behind the network
  examples.
