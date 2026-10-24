BrainPy-style Modeling
======================

The BrainPy-style layer is the native modeling surface of ``brainpy.state``: a
library of composable spiking neurons, synapses, synaptic outputs, projections,
short-term plasticity, and readouts that you assemble into networks and drive
with the ``brainstate`` transform primitives. The same building blocks serve
**both** worlds — biophysical simulation *and* gradient-based, brain-inspired
training — because every model is a differentiable, state-based ``Module``.

How to use this guide
---------------------

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Start with the tutorials
      :link: tutorials/01-first-neuron
      :link-type: doc

      Four sequential, learning-oriented notebooks that take you from a single
      neuron to a trained spiking network. Read them in order the first time.

   .. grid-item-card:: Reach for how-to by task
      :link: howto/index
      :link-type: doc

      Goal-directed recipes, split into a **Simulation** track and a
      **Training** track. Jump straight to the task you have in front of you.

   .. grid-item-card:: Look up models in the API reference
      :link: /apis/state
      :link-type: doc

      There is no separate model catalog: the autodoc API reference *is* the
      directory. Browse neurons, synapses, projections, and outputs by family,
      with full signatures and docstrings.

Tutorials vs. how-to
--------------------

The two sections answer different questions, in the spirit of the
`Diátaxis <https://diataxis.fr/>`_ framework:

- **Tutorials** are a guided lesson. They are sequential, they assume no prior
  knowledge of the API, and each one builds on the last. Their job is to give
  you a working mental model of how neurons, synapses, projections, and the
  transform loops fit together.
- **How-to guides** are task recipes. They assume you already know the basics
  and want to accomplish one concrete thing — choose a neuron model, swap a
  synaptic output, add a transmission delay, train through a long rollout
  without exhausting memory. Each is independent; read only the one you need.

For the *why* behind the design — the State paradigm, physical units, and the
AlignPre/AlignPost projection scheme that ties simulation and learning together
— see the :doc:`Core Concepts </state/concepts/index>` spine. Tutorials and how-to
guides link back to the relevant concept chapters as they go.

.. toctree::
   :hidden:
   :caption: Get started

   5-minute-tour
   mental-model

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/01-first-neuron
   tutorials/02-synapse-and-projection
   tutorials/03-ei-balanced-network
   tutorials/04-train-an-snn

.. toctree::
   :hidden:
   :caption: How-to guides

   howto/index
