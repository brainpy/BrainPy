Synapse Models
==============

Synaptic dynamics models in BrainPy.

.. currentmodule:: brainpy

Base Class
----------

Synapse
~~~~~~~

.. class:: Synapse(size, **kwargs)

   Base class for all synapse models.

   **Parameters:**

   - ``size`` (int) - Number of post-synaptic neurons
   - ``**kwargs`` - Additional keyword arguments

   **Key Methods:**

   .. method:: update(x)

      Update synaptic dynamics.

      :param x: Pre-synaptic input (typically spike indicator)
      :returns: Synaptic conductance/current

   .. method:: reset_state(batch_size=None)

      Reset synaptic state.

   **Descriptor Pattern:**

   Synapses use the ``.desc()`` class method for use in projections:

   .. code-block:: python

      syn = bp.Expon.desc(size=100, tau=5*u.ms)

Simple Synapses
---------------

Delta
~~~~~

.. class:: Delta

   Instantaneous synaptic transmission (no dynamics).

   .. math::

      g(t) = \\sum_k \\delta(t - t_k)

   **Usage:**

   .. code-block:: python

      syn = bp.Delta.desc(100)

Expon
~~~~~

.. class:: Expon

   Exponential synapse (single time constant).

   .. math::

      \\tau \\frac{dg}{dt} = -g + \\sum_k \\delta(t - t_k)

   **Parameters:**

   - ``size`` (int) - Population size
   - ``tau`` (Quantity[ms]) - Time constant (default: 5 ms)

   **Example:**

   .. code-block:: python

      syn = bp.Expon.desc(size=100, tau=5*u.ms)

Alpha
~~~~~

.. class:: Alpha

   Alpha function synapse (rise + decay).

   .. math::

      \\tau \\frac{dg}{dt} &= -g + h \\\\
      \\tau \\frac{dh}{dt} &= -h + \\sum_k \\delta(t - t_k)

   **Parameters:**

   - ``size`` (int) - Population size
   - ``tau`` (Quantity[ms]) - Characteristic time (default: 10 ms)

   **Example:**

   .. code-block:: python

      syn = bp.Alpha.desc(size=100, tau=10*u.ms)

DualExponential
~~~~~~~~~~~~~~~

.. class:: DualExponential

   Biexponential synapse with separate rise/decay.

   **Parameters:**

   - ``size`` (int) - Population size
   - ``tau_rise`` (Quantity[ms]) - Rise time constant
   - ``tau_decay`` (Quantity[ms]) - Decay time constant

Receptor Models
---------------

AMPA
~~~~

.. class:: AMPA

   AMPA receptor (fast excitatory).

   **Parameters:**

   - ``size`` (int) - Population size
   - ``tau`` (Quantity[ms]) - Time constant (default: 2 ms)

   **Example:**

   .. code-block:: python

      syn = bp.AMPA.desc(size=100, tau=2*u.ms)

NMDA
~~~~

.. class:: NMDA

   NMDA receptor (slow excitatory, voltage-dependent).

   **Parameters:**

   - ``size`` (int) - Population size
   - ``tau_rise`` (Quantity[ms]) - Rise time (default: 2 ms)
   - ``tau_decay`` (Quantity[ms]) - Decay time (default: 100 ms)
   - ``a`` (Quantity[1/mM]) - Mg²⁺ sensitivity (default: 0.5/mM)
   - ``cc_Mg`` (Quantity[mM]) - Mg²⁺ concentration (default: 1.2 mM)

   **Example:**

   .. code-block:: python

      syn = bp.NMDA.desc(
          size=100,
          tau_rise=2*u.ms,
          tau_decay=100*u.ms
      )

GABAa
~~~~~

.. class:: GABAa

   GABA_A receptor (fast inhibitory).

   **Parameters:**

   - ``size`` (int) - Population size
   - ``tau`` (Quantity[ms]) - Time constant (default: 6 ms)

   **Example:**

   .. code-block:: python

      syn = bp.GABAa.desc(size=100, tau=6*u.ms)

GABAb
~~~~~

.. class:: GABAb

   GABA_B receptor (slow inhibitory).

   **Parameters:**

   - ``size`` (int) - Population size
   - ``tau_rise`` (Quantity[ms]) - Rise time (default: 3.5 ms)
   - ``tau_decay`` (Quantity[ms]) - Decay time (default: 150 ms)

Short-Term Plasticity
---------------------

STD
~~~

.. class:: STD

   Short-term depression.

   **Parameters:**

   - ``size`` (int) - Population size
   - ``tau`` (Quantity[ms]) - Synaptic time constant
   - ``tau_d`` (Quantity[ms]) - Depression recovery time
   - ``U`` (float) - Utilization fraction (0-1)

STF
~~~

.. class:: STF

   Short-term facilitation.

   **Parameters:**

   - ``size`` (int) - Population size
   - ``tau`` (Quantity[ms]) - Synaptic time constant
   - ``tau_f`` (Quantity[ms]) - Facilitation time constant
   - ``U`` (float) - Baseline utilization

STP
~~~

.. class:: STP

   Combined short-term plasticity (depression + facilitation).

   **Parameters:**

   - ``size`` (int) - Population size
   - ``tau`` (Quantity[ms]) - Synaptic time constant
   - ``tau_d`` (Quantity[ms]) - Depression time constant
   - ``tau_f`` (Quantity[ms]) - Facilitation time constant
   - ``U`` (float) - Baseline utilization

Output Models
-------------

CUBA
~~~~

.. class:: CUBA

   Current-based synaptic output.

   .. math::

      I_{syn} = g_{syn}

   **Usage:**

   .. code-block:: python

      out = bp.CUBA.desc()

COBA
~~~~

.. class:: COBA

   Conductance-based synaptic output.

   .. math::

      I_{syn} = g_{syn} (E_{syn} - V_{post})

   **Parameters:**

   - ``E`` (Quantity[mV]) - Reversal potential

   **Example:**

   .. code-block:: python

      # Excitatory
      out_exc = bp.COBA.desc(E=0*u.mV)

      # Inhibitory
      out_inh = bp.COBA.desc(E=-80*u.mV)

MgBlock
~~~~~~~

.. class:: MgBlock

   Voltage-dependent magnesium block (for NMDA).

   **Parameters:**

   - ``E`` (Quantity[mV]) - Reversal potential
   - ``cc_Mg`` (Quantity[mM]) - Mg²⁺ concentration
   - ``alpha`` (Quantity[1/mV]) - Voltage sensitivity
   - ``beta`` (float) - Voltage offset

Usage in Projections
---------------------

**Standard pattern:**

.. code-block:: python

   proj = bp.AlignPostProj(
       comm=brainstate.nn.EventFixedProb(n_pre, n_post, prob=0.1, weight=0.5*u.mS),
       syn=bp.Expon.desc(n_post, tau=5*u.ms),  # Synapse
       out=bp.COBA.desc(E=0*u.mV),              # Output
       post=post_neurons
   )

**Receptor-specific:**

.. code-block:: python

   # Fast excitation (AMPA)
   ampa_proj = bp.AlignPostProj(
       comm=...,
       syn=bp.AMPA.desc(n_post, tau=2*u.ms),
       out=bp.COBA.desc(E=0*u.mV),
       post=post_neurons
   )

   # Slow excitation (NMDA)
   nmda_proj = bp.AlignPostProj(
       comm=...,
       syn=bp.NMDA.desc(n_post, tau_decay=100*u.ms),
       out=bp.MgBlock.desc(E=0*u.mV),
       post=post_neurons
   )

   # Fast inhibition (GABA_A)
   gaba_proj = bp.AlignPostProj(
       comm=...,
       syn=bp.GABAa.desc(n_post, tau=6*u.ms),
       out=bp.COBA.desc(E=-80*u.mV),
       post=post_neurons
   )

See Also
--------

- :doc:`../core-concepts/synapses` - Detailed synapse guide
- :doc:`../tutorials/basic/02-synapse-models` - Synapse tutorial
- :doc:`../tutorials/advanced/06-synaptic-plasticity` - Plasticity tutorial
- :doc:`projections` - Projection API
