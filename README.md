<p align="center">
  	<img alt="Header image of BrainPy - brain dynamics programming in Python." src="./images/logo.png"  >
</p> 



<p align="center">
	<a href="https://github.com/PKU-NIP-Lab/BrainPy"><img alt="LICENSE" src="https://anaconda.org/brainpy/brainpy/badges/license.svg"></a>
  	<a href="https://brainpy.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation" src="https://readthedocs.org/projects/brainpy/badge/?version=latest"></a>
  	<a href="https://badge.fury.io/py/brain-py"><img alt="PyPI version" src="https://badge.fury.io/py/brain-py.svg"></a>
  	<a href="https://travis-ci.com/PKU-NIP-Lab/BrainPy"><img alt="Build Status" src="https://travis-ci.com/PKU-NIP-Lab/BrainPy.svg?branch=master"></a>
</p>



:clap::clap: **CHEERS**: A new version of BrainPy (>=2.0.0) has been released! :clap::clap: 



# Why use BrainPy

``BrainPy`` is an integrative framework for computational neuroscience and brain-inspired computation based on the Just-In-Time (JIT) compilation (built on top of [JAX](https://github.com/google/jax)). Core functions provided in BrainPy includes

- **JIT compilation** for class objects. 
- **Numerical solvers** for ODEs, SDEs, and others. 
- **Dynamics simulation tools** for various brain objects, like neurons, synapses, networks, soma, dendrites, channels, and even more. 
- **Dynamics analysis tools** for differential equations, including phase plane analysis and bifurcation analysis, and linearization analysis.
- **Seamless integration with deep learning models**.
- And more ......

`BrainPy` is designed to effectively satisfy your basic requirements: 

- **Pythonic**: BrainPy is based on Python language and has a Pythonic coding style. 
- **Flexible and transparent**: BrainPy endows the users with full data/logic flow control. Users can code any logic they want with BrainPy. 
- **Extensible**: BrainPy allows users to extend new functionality just based on Python code. Almost every part of the BrainPy system can be extended to be customized. 
- **Efficient**: All codes in BrainPy can be just-in-time compiled (based on [JAX](https://github.com/google/jax)) to run on CPU, GPU, or TPU devices, thus guaranteeing its running efficiency. 



# How to use BrainPy

## Step 1: installation

``BrainPy`` is based on Python (>=3.6), and the following packages are required to be installed to use ``BrainPy``: `numpy >= 1.15`, `matplotlib >= 3.4`, and `jax >= 0.2.10` ([how to install jax?](https://brainpy.readthedocs.io/en/latest/quickstart/installation.html#dependency-2-jax))

``BrainPy`` can be installed on  Linux (Ubuntu 16.04 or later), macOS (10.12 or later), and Windows platforms. Use the following instructions to install ``brainpy``:

```bash
pip install brain-py -U
```

*For the full installation details please see documentation: [Quickstart/Installation](https://brainpy.readthedocs.io/en/latest/quickstart/installation.html)*




## Step 2: useful links

- **Documentation:** https://brainpy.readthedocs.io/
- **Bug reports:** https://github.com/PKU-NIP-Lab/BrainPy/issues
- **Examples from papers**: https://brainpy-examples.readthedocs.io/
- **Canonical brain models**: https://brainmodels.readthedocs.io/



## Step 3: inspirational examples

Here we list several examples of BrainPy. For more detailed examples and tutorials please see [**BrainModels**](https://brainmodels.readthedocs.io) or [**BrainPy-Examples**](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/). 



### Neuron models

- [Leaky integrate-and-fire neuron model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.neurons.LIF.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/neurons/LIF.py)
- [Exponential integrate-and-fire neuron model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.neurons.ExpIF.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/neurons/ExpIF.py)
- [Quadratic integrate-and-fire neuron model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.neurons.QuaIF.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/neurons/QuaIF.py)
- [Adaptive Quadratic integrate-and-fire model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.neurons.AdQuaIF.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/neurons/AdQuaIF.py)
- [Adaptive Exponential integrate-and-fire model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.neurons.AdExIF.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/neurons/AdExIF.py)
- [Generalized integrate-and-fire model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.neurons.GIF.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/neurons/GIF.py)
- [Hodgkin–Huxley neuron model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.neurons.HH.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/neurons/HH.py)
- [Izhikevich neuron model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.neurons.Izhikevich.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/neurons/Izhikevich.py)
- [Morris-Lecar neuron model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.neurons.MorrisLecar.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/neurons/MorrisLecar.py)
- [Hindmarsh-Rose bursting neuron model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.neurons.HindmarshRose.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/neurons/HindmarshRose.py)

See [brainmodels.neurons](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/neurons.html) to find more.



### Synapse models

- [Voltage jump synapse model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.synapses.VoltageJump.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/synapses/voltage_jump.py)
- [Exponential synapse model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.synapses.ExpCUBA.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/synapses/exponential.py)
- [Alpha synapse model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.synapses.AlphaCUBA.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/synapses/alpha.py)
- [Dual exponential synapse model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.synapses.DualExpCUBA.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/synapses/dual_exp.py)
- [AMPA synapse model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.synapses.AMPA.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/synapses/AMPA.py)
- [GABAA synapse model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.synapses.GABAa.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/synapses/GABAa.py)
- [NMDA synapse model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.synapses.NMDA.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/synapses/NMDA.py)
- [Short-term plasticity model](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/generated/brainmodels.synapses.STP.html), [source code](https://github.com/PKU-NIP-Lab/BrainModels/blob/brainpy-2.x/brainmodels/synapses/STP.py)

See [brainmodels.synapses](https://brainmodels.readthedocs.io/en/brainpy-2.x/apis/synapses.html) to find more.



### Network models

- **[CANN]** [*(Si Wu, 2008)* Continuous-attractor Neural Network](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/cann/Wu_2008_CANN.html)
- [*(Vreeswijk & Sompolinsky, 1996)* E/I balanced network](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/ei_nets/Vreeswijk_1996_EI_net.html)
- [*(Sherman & Rinzel, 1992)* Gap junction leads to anti-synchronization](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/gj_nets/Sherman_1992_gj_antisynchrony.html)
- [*(Wang & Buzsáki, 1996)* Gamma Oscillation](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/oscillation_synchronization/Wang_1996_gamma_oscillation.html)
- [*(Brunel & Hakim, 1999)* Fast Global Oscillation](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/oscillation_synchronization/Brunel_Hakim_1999_fast_oscillation.html)
- [*(Diesmann, et, al., 1999)* Synfire Chains](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/oscillation_synchronization/Diesmann_1999_synfire_chains.html)
- **[Working Memory]** [*(Mi, et. al., 2017)* STP for Working Memory Capacity](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/working_memory/Mi_2017_working_memory_capacity.html)
- **[Working Memory]** [*(Bouchacourt & Buschman, 2019)* Flexible Working Memory Model](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/working_memory/Bouchacourt_2019_Flexible_working_memory.html)
- **[Decision Making]** [*(Wang, 2002)* Decision making spiking model](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/decision_making/Wang_2002_decision_making_spiking.html)



### Dynamics training

- [Train Integrator RNN with BP](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/recurrent_networks/integrator_rnn.html)

- [*(Sussillo & Abbott, 2009)* FORCE Learning](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/recurrent_networks/Sussillo_Abbott_2009_FORCE_Learning.html)

- [*(Laje & Buonomano, 2013)* Robust Timing in RNN](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/recurrent_networks/Laje_Buonomano_2013_robust_timing_rnn.html)
- [*(Song, et al., 2016)*: Training excitatory-inhibitory recurrent network](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/recurrent_networks/Song_2016_EI_RNN.html)
- **[Working Memory]** [*(Masse, et al., 2019)*: RNN with STP for Working Memory](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/recurrent_networks/Masse_2019_STP_RNN.html)




### Low-dimensional dynamics analysis

- [[1D] Simple systems](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/dynamics_analysis/1d_simple_systems.html)
- [[2D] NaK model analysis](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/dynamics_analysis/2d_NaK_model.html)
- [[3D] Hindmarsh Rose Model](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/dynamics_analysis/3d_hindmarsh_rose_model.html)
- **[Decision Making Model]** [[2D] Decision making rate model](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/decision_making/Wang_2006_decision_making_rate.html)



### High-dimensional dynamics analysis

- [*(Yang, 2020)*: Dynamical system analysis for RNN](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/recurrent_networks/Yang_2020_RNN_Analysis.html)

- [Continuous-attractor Neural Network](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/dynamics_analysis/highdim_CANN.html)
- [Gap junction-coupled FitzHugh-Nagumo Model](https://brainpy-examples.readthedocs.io/en/brainpy-2.x/dynamics_analysis/highdim_gj_coupled_fhn.html)





# BrainPy V1

If you are using ``brainpy==1.x``, you can find *documentation*, *examples*, and *models* through the following links:

- **Documentation:** https://brainpy.readthedocs.io/en/brainpy-1.x/
- **Examples from papers**: https://brainpy-examples.readthedocs.io/en/brainpy-1.x/
- **Canonical brain models**: https://brainmodels.readthedocs.io/en/brainpy-1.x/

The changes from ``brainpy==1.x`` to ``brainpy==2.x`` can be inspected through [API documentation: release notes](https://brainpy.readthedocs.io/en/latest/apis/changelog.html).





