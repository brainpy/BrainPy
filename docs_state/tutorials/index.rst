Tutorials
=========

Welcome to the ``brainpy.state`` tutorials! These step-by-step guides will help you master computational neuroscience modeling with BrainPy.

Learning Path
-------------

We recommend following the tutorials in order:

1. **Basic Tutorials**: Learn core components (neurons, synapses, networks)
2. **Advanced Tutorials**: Master complex topics (training, plasticity, large-scale simulations)
3. **Specialized Topics**: Explore specific applications and techniques

Basic Tutorials
---------------

Start here to learn the fundamentals of ``brainpy.state``.

Tutorial 1: LIF Neuron Basics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn the most important spiking neuron model.

**Topics covered:**

- Creating and configuring LIF neurons
- Simulating neuron dynamics
- Computing F-I curves
- Hard vs soft reset modes
- Working with neuron populations
- Parameter effects on behavior

:doc:`Go to Tutorial 1 <basic/01-lif-neuron>`

**Prerequisites:** None (start here!)

**Duration:** ~30 minutes

---

Tutorial 2: Synapse Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Understand temporal filtering and synaptic dynamics.

**Topics covered:**

- Exponential synapses
- Alpha synapses
- AMPA and GABA receptors
- Comparing synapse models
- Custom synapse creation

:doc:`Go to Tutorial 2 <basic/02-synapse-models>`

**Prerequisites:** Tutorial 1

**Duration:** ~25 minutes

---

Tutorial 3: Network Connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build connected neural networks.

**Topics covered:**

- Projection architecture (Comm-Syn-Out)
- Fixed probability connectivity
- CUBA vs COBA synapses
- E-I balanced networks
- Network visualization

:doc:`Go to Tutorial 3 <basic/03-network-connections>`

**Prerequisites:** Tutorials 1-2

**Duration:** ~35 minutes

---

Tutorial 4: Input and Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate inputs and process network outputs.

**Topics covered:**

- Poisson spike trains
- Periodic inputs
- Custom input patterns
- Readout layers
- Population coding

:doc:`Go to Tutorial 4 <basic/04-input-output>`

**Prerequisites:** Tutorials 1-3

**Duration:** ~20 minutes

Advanced Tutorials
------------------

Dive deeper into sophisticated modeling techniques.

Tutorial 5: Training Spiking Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learn gradient-based training for SNNs.

**Topics covered:**

- Surrogate gradient methods
- BPTT for SNNs
- Loss functions for spikes
- Optimizers and learning rates
- Classification tasks
- Training loops

:doc:`Go to Tutorial 5 <advanced/05-snn-training>`

**Prerequisites:** Basic Tutorials 1-4

**Duration:** ~45 minutes

---

Tutorial 6: Synaptic Plasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement learning rules and adaptation.

**Topics covered:**

- Short-term plasticity (STP)
- Depression and facilitation
- STDP principles
- Homeostatic mechanisms
- Network learning

:doc:`Go to Tutorial 6 <advanced/06-synaptic-plasticity>`

**Prerequisites:** Basic Tutorials, Tutorial 5

**Duration:** ~40 minutes

---

Tutorial 7: Large-Scale Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scale up your models efficiently.

**Topics covered:**

- Memory optimization
- JIT compilation best practices
- Batching strategies
- GPU/TPU acceleration
- Performance profiling
- Sparse connectivity

:doc:`Go to Tutorial 7 <advanced/07-large-scale-simulations>`

**Prerequisites:** All Basic Tutorials

**Duration:** ~35 minutes

Specialized Topics
------------------

Application-specific tutorials for advanced users.

Brain Oscillations (Coming Soon)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model and analyze network rhythms.

**Topics:** Gamma oscillations, synchrony, oscillation mechanisms

**Prerequisites:** Advanced

---

Decision-Making Networks (Coming Soon)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build cognitive computation models.

**Topics:** Attractor dynamics, competition, working memory

**Prerequisites:** Advanced

---

Reservoir Computing (Coming Soon)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use untrained recurrent networks for computation.

**Topics:** Echo state networks, liquid state machines, readout training

**Prerequisites:** Advanced

Tutorial Format
---------------

Each tutorial includes:

✅ **Clear learning objectives**: Know what you'll learn

✅ **Runnable code**: All examples work out of the box

✅ **Visualizations**: See your models in action

✅ **Explanations**: Understand the "why" behind the code

✅ **Exercises**: Practice what you've learned

✅ **References**: Links to papers and further reading

How to Use These Tutorials
---------------------------

Interactive (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

Download and run the Jupyter notebooks:

.. code-block:: bash

    git clone https://github.com/brainpy/BrainPy.git
    cd BrainPy/docs_version3/tutorials/basic
    jupyter notebook 01-lif-neuron.ipynb

Read-Only
~~~~~~~~~

Browse the tutorials online in the documentation.

Binder
~~~~~~

Run tutorials in your browser without installation:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/brainpy/BrainPy-binder/main
   :alt: Binder

Prerequisites
-------------

Before starting the tutorials, ensure you have:

✅ Python 3.10 or later

✅ ``brainpy.state`` installed (see :doc:`../quickstart/installation`)

✅ Basic Python knowledge (functions, classes, NumPy)

✅ Basic neuroscience concepts (optional but helpful)

Recommended setup:

.. code-block:: bash

    pip install brainpy[cpu] matplotlib jupyter -U

Additional Resources
--------------------

**For Quick Start**
   See the :doc:`../quickstart/5min-tutorial` for a rapid introduction

**For Concepts**
   Read :doc:`../core-concepts/architecture` for architectural understanding

**For Examples**
   Browse :doc:`../examples/gallery` for complete, real-world models

**For Reference**
   Consult the :doc:`../apis` for detailed API documentation

Getting Help
------------

If you get stuck:

- Check the :doc:`FAQ <../migration/migration-guide>` (Migration Guide has troubleshooting)
- Search `GitHub Issues <https://github.com/brainpy/BrainPy/issues>`_
- Ask on GitHub Discussions
- Review the `brainstate documentation <https://brainstate.readthedocs.io/>`_

Tutorial Roadmap
----------------

**Currently Available:**

**Basic Tutorials:**
- ✅ Tutorial 1: LIF Neuron Basics
- ✅ Tutorial 2: Synapse Models
- ✅ Tutorial 3: Network Connections
- ✅ Tutorial 4: Input and Output

**Advanced Tutorials:**
- ✅ Tutorial 5: Training SNNs
- ✅ Tutorial 6: Synaptic Plasticity
- ✅ Tutorial 7: Large-Scale Simulations

**Future Plans:**

- Brain Oscillations
- Decision-Making Networks
- Reservoir Computing
- Custom Components
- Advanced Training Techniques

We're actively developing new tutorials. Star the repository to stay updated!

Contributing
------------

Want to contribute a tutorial? We'd love your help!

1. Check the `contribution guidelines <https://github.com/brainpy/BrainPy/blob/master/CONTRIBUTING.md>`_
2. Open an issue to discuss your tutorial idea
3. Submit a pull request

Good tutorial topics:

- Specific neuron models (Izhikevich, AdEx, etc.)
- Network architectures (attractor networks, etc.)
- Analysis techniques (spike train analysis, etc.)
- Applications (sensory processing, motor control, etc.)

Let's Start!
------------

Ready to begin? Start with :doc:`Tutorial 1: LIF Neuron Basics <basic/01-lif-neuron>`!


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Basic Tutorials

   basic/01-lif-neuron.ipynb
   basic/02-synapse-models.ipynb
   basic/03-network-connections.ipynb
   basic/04-input-output.ipynb


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Advanced Tutorials

   advanced/05-snn-training.ipynb
   advanced/06-synaptic-plasticity.ipynb
   advanced/07-large-scale-simulations.ipynb
