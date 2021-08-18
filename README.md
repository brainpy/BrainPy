
![Logo](docs/_static/logo.png)

[![LICENSE](https://anaconda.org/brainpy/brainpy/badges/license.svg)](https://github.com/PKU-NIP-Lab/BrainPy)    [![Documentation](https://readthedocs.org/projects/brainpy/badge/?version=latest)](https://brainpy.readthedocs.io/en/latest/?badge=latest)     [![Conda](https://anaconda.org/brainpy/brainpy-simulator/badges/version.svg)](https://anaconda.org/brainpy/brainpy-simulator)  [![PyPI version](https://badge.fury.io/py/brainpy-simulator.svg)](https://badge.fury.io/py/brainpy-simulator) [![Build Status](https://travis-ci.com/PKU-NIP-Lab/BrainPy.svg?branch=master)](https://travis-ci.com/PKU-NIP-Lab/BrainPy)



# Why to use BrainPy

``BrainPy`` is an integrative framework for computational neuroscience and brain-inspired computation based on Just-In-Time (JIT) compilation. Core functions provided in BrainPy includes

1. **General numerical solvers** for ODEs, SDEs, DDEs, FDEs, and others.

2. **Dynamics simulation tools** for various brain objects, like neurons, synapses, networks, soma, dendrites, channels, and even molecular.

3. **Dynamics analysis tools** for differential equations, including phase plane analysis and bifurcation analysis, continuation analysis and sensitive analysis.

4. **Seamless integration with deep learning models**, and has the speed benefit on JIT compilation.

Moreover, `BrainPy` is designed to effectively satisfy your basic requirements: 

- *Easy to learn and use*, because BrainPy is only based on Python language and has little dependency requirements; 
- *Highly flexible and transparent*, because BrainPy endows the users with the fully data/logic flow control; 
- *Simulation can be guided with the analysis*, because the same code in BrainPy can not only be used for simulation, but also for dynamics analysis; 
- *Efficient running speed*, because BrainPy is designed to compile your codes just-in-time.

Currently, `BrainPy` heavily relies on the JIT compilers [Numba](https://numba.pydata.org/) and [JAX](https://jax.readthedocs.io/) on CPU or GPU devices. Extending BrainPy to support other backend frameworks you prefer is also easy. The details please see documents coming soon. 



# How to use BrainPy

## Step 1: installation

``BrainPy`` is based on Python (>=3.6), and the following packages are required to be installed to use ``BrainPy``:

- NumPy >= 1.15
- Matplotlib >= 3.3

**Method 1**: install ``BrainPy`` by using ``pip``:

```bash
> pip install -U brainpy-simulator
```

**Method 2**: install ``BrainPy`` by using ``conda``:

```bash
> conda install brainpy-simulator -c brainpy
```

**Method 3**: install ``BrainPy`` from source:

```bash
> pip install git+https://github.com/PKU-NIP-Lab/BrainPy
> # or
> pip install git+https://git.openi.org.cn/OpenI/BrainPy
> # or
> pip install -e git://github.com/PKU-NIP-Lab/BrainPy.git@V1.0.0
```



**Other dependencies**: you want to get the full supports by BrainPy, please install the following packages:

- `JAX >= 0.2.10`,  needed for "jax" backend and "dnn" module
- `Numba >= 0.52`,  needed for JIT compilation on "numpy" backend
- `SymPy >= 1.4`, needed for dynamics "analysis" module and Exponential Euler method



## Step 2: useful links

- **Documentation:** https://brainpy.readthedocs.io/
- **Source code:** https://github.com/PKU-NIP-Lab/BrainPy   or   https://git.openi.org.cn/OpenI/BrainPy
- **Bug reports:** https://github.com/PKU-NIP-Lab/BrainPy/issues   or   Email to adaduo@outlook.com
- **Examples from papers**: https://brainmodels.readthedocs.io/en/latest/from_papers.html



## Step 3: comprehensive examples

Here list several simple examples for dynamics simulation and analysis. Comprehensive examples and tutorials please see [BrainModels](https://brainmodels.readthedocs.io).

### Dynamics simulation

- [Hodgkin–Huxley neuron model](https://github.com/PKU-NIP-Lab/BrainModels/blob/main/brainmodels/tensor_backend/neurons/HodgkinHuxley_model.py)
- [AMPA synapse model](https://github.com/PKU-NIP-Lab/BrainModels/blob/main/brainmodels/tensor_backend/synapses/AMPA_synapse.py)
- [Gamma oscillation network model](https://brainmodels.readthedocs.io/en/latest/from_papers/Wang_1996_gamma_oscillation.html)
- [E/I balanced network model](https://brainmodels.readthedocs.io/en/latest/from_papers/Vreeswijk_1996_EI_net.html)
- [Continuous attractor network model](https://brainmodels.readthedocs.io/en/latest/from_papers/Wu_2008_CANN.html)


### Dynamics analysis

- [Phase plane analysis of the I<sub>Na,p</sub>-I<sub>K</sub> model](https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/NaK_model_analysis.html)
- [Codimension 1 bifurcation analysis of FitzHugh Nagumo model](https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/FitzHugh_Nagumo_analysis.html)
- [Codimension 2 bifurcation analysis of FitzHugh Nagumo model](https://brainmodels.readthedocs.io/en/latest/tutorials/dynamics_analysis/FitzHugh_Nagumo_analysis.html#Codimension-2-bifurcation-analysis)

### Deep neural networks



