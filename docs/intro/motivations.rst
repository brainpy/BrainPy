Motivation
==========

A variety of Python SNN simulators are available in the internet, such as
`Brain2 <https://github.com/brian-team/brian2>`_,
`ANNarchy <https://github.com/ANNarchy/ANNarchy>`_,
`NEST <http://www.nest-initiative.org/>`_, etc.
However, almost all of them are using the `code generation` approach. That is to say, the
essence of these framework is let you use python scripts to control the writing of
c/c++ codes. The advantage of these frameworks is obvious: they provide the easiest way
to define the model (by using high-level descriptive language `python`), at the same time,
get the fast run-time speed in the low level language (by using the backend `c++` code).
However, several drawbacks also exist:

- Any `code generation` framework has its own **fixed** templates to generate backend c++ codes.
  However, there will always be exceptions beyond the framework, such as the data or logical
  flows that the framework do not consider before. Therefore, the discrepancy emerges:
  If you want to generate highly efficient low-level language codes, you must provide a
  fixed code-generation template for high-level descriptions; Once, if you have a logic control
  beyond the template, you must want to extend this template. However, the extension of
  the framework is not a easy thing for the general users (even for mature users).
- Meanwhile, no framework is immune to errors. In `Brain2` and `ANNarchy`, some models are
  wrongly coded and users are hard to correct them,
  such as the `gap junction model for leaky integrate-and-fire neurons` in `Brian2`
  (see :doc:`gapjunction_lif_in_brian2 <gapjunction_lif_in_brian2>`),
  `Hodgkin–Huxley neuron model` in `ANNarchy` (see :doc:`HH_model_in_ANNarchy <HH_model_in_ANNarchy>`).
  These facts further point out that we need a framework that is friendly and easy
  for user-defines.
- Moreover, not all SNN simulations require the c++ acceleration. In `code generation` framework,
  too much times are spent in the compilation of generated c++ codes. However, for small
  network simulations, the running time is usually lower than that compilation time. Thus, the
  native NumPy codes (many functions are also written in c++) are much faster than the `so called`
  accelerated codes.
- Finally, just because of highly dependence on code generation, a lot of garbage (such as
  the compiled files and the link files) is left after code running, and users are hard to
  debug the defined models, making the model coding much more limited and difficult.

Therefore, ``NumpyBrain`` wants to provide a highly flexible SNN simulation framework for
Python users. It endows the users with the fully data/logic flow control. The core of the
framework is a micro-kernel, and it's easy to understand. Based on the kernel, the extension
of the new models or the customization of the data/logic flows are very simple for users.
Ample examples (such as LIF neuron, HH neuron, or AMPA synapse, GABA synapse and GapJunction) are also
provided. More details please see the `document <https://numpybrain.readthedocs.io/en/latest/>`_.

The design of ``NumpyBrain`` is guided by the following principles:

- **Modularity**. A network can be broken down into various ``neurons`` and ``synapses``.
  To inspect the inner dynamical structure of these elements, we need the ``Monitor`` to
  record the running trajectory for each object. In ``NumpyBrain``, there are only these
  three kinds of objects. Such objects can be plugged together almost arbitrarily (only
  with few restrictions) to form a new network.
- **Easy extensibility**. For each kind of object, new models (neurons or synapses) are
  simple to add, and existing models provide ample examples.
- **User friendliness**. The data flow in each object is transparent, and can be easily
  controlled by users. Users can define or modify the data or logical flow by themselves
  according to need.
- **Plug and play**. No garbage file will be generated and left after any code-running.
  Just, use or not use.


