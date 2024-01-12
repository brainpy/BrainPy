<p align="center">
  	<img alt="Header image of BrainPy - brain dynamics programming in Python." src="https://github.com/brainpy/BrainPy/blob/master/images/logo.png" width=80%>
</p> 



<p align="center">
	<a href="https://pypi.org/project/brainpy/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainpy"></a>
	<a href="https://github.com/brainpy/BrainPy"><img alt="LICENSE" src="https://anaconda.org/brainpy/brainpy/badges/license.svg"></a>
  	<a href="https://brainpy.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation" src="https://readthedocs.org/projects/brainpy/badge/?version=latest"></a>
  	<a href="https://badge.fury.io/py/brainpy"><img alt="PyPI version" src="https://badge.fury.io/py/brainpy.svg"></a>
    <a href="https://github.com/brainpy/BrainPy/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/brainpy/BrainPy/actions/workflows/CI.yml/badge.svg"></a>
    <a href="https://github.com/brainpy/BrainPy/actions/workflows/CI-models.yml"><img alt="Continuous Integration with Models" src="https://github.com/brainpy/BrainPy/actions/workflows/CI-models.yml/badge.svg"></a>
</p>


BrainPy is a flexible, efficient, and extensible framework for computational neuroscience and brain-inspired computation based on the Just-In-Time (JIT) compilation (built on top of [JAX](https://github.com/google/jax), [Taichi](https://github.com/taichi-dev/taichi), [Numba](https://github.com/numba/numba), and others). It provides an integrative ecosystem for brain dynamics programming, including brain dynamics **building**, **simulation**, **training**, **analysis**, etc. 

- **Website (documentation and APIs)**: https://brainpy.readthedocs.io/en/latest
- **Source**: https://github.com/brainpy/BrainPy
- **Bug reports**: https://github.com/brainpy/BrainPy/issues
- **Source on OpenI**: https://git.openi.org.cn/OpenI/BrainPy



## Installation

BrainPy is based on Python (>=3.8) and can be installed on Linux (Ubuntu 16.04 or later), macOS (10.12 or later), and Windows platforms. Install the latest version of BrainPy:

```bash
$ pip install brainpy -U
```

In addition, many customized operators in BrainPy are implemented in ``brainpylib``.
Install the latest version of `brainpylib` by:

```bash
# CPU installation for Linux, macOS and Windows
$ pip install --upgrade brainpylib
```

```bash
# CUDA 12 installation for Linux only
$ pip install --upgrade brainpylib-cu12x
```

```bash
# CUDA 11 installation for Linux only
$ pip install --upgrade brainpylib-cu11x
```

For detailed installation instructions, please refer to the documentation: [Quickstart/Installation](https://brainpy.readthedocs.io/en/latest/quickstart/installation.html)


### Using BrainPy with docker

We provide a docker image for BrainPy. You can use the following command to pull the image:
```bash
$ docker pull brainpy/brainpy:latest
```

Then, you can run the image with the following command:
```bash
$ docker run -it --platform linux/amd64 brainpy/brainpy:latest
```


### Using BrainPy with Binder

We provide a Binder environment for BrainPy. You can use the following button to launch the environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/brainpy/BrainPy-binder/main)


## Ecosystem

- **[BrainPy](https://github.com/brainpy/BrainPy)**: The solution for the general-purpose brain dynamics programming. 
- **[brainpy-examples](https://github.com/brainpy/examples)**: Comprehensive examples of BrainPy computation. 
- **[brainpy-datasets](https://github.com/brainpy/datasets)**: Neuromorphic and Cognitive Datasets for Brain Dynamics Modeling.
- [《神经计算建模实战》 (Neural Modeling in Action)](https://github.com/c-xy17/NeuralModeling)
- [第一届神经计算建模与编程培训班 (First Training Course on Neural Modeling and Programming)](https://github.com/brainpy/1st-neural-modeling-and-programming-course)
- [第二届神经计算建模与编程培训班 (Second Training Course on Neural Modeling and Programming)](https://github.com/brainpy/2nd-neural-modeling-and-programming-course)


## Citing 

BrainPy is developed by a team in Neural Information Processing Lab at Peking University, China. 
Our team is committed to the long-term maintenance and development of the project. 

If you are using ``brainpy``, please consider citing [the corresponding papers](https://brainpy.readthedocs.io/en/latest/tutorial_FAQs/citing_and_publication.html). 



## Ongoing development plans

We highlight the key features and functionalities that are currently under active development. 

We also welcome your contributions 
(see [Contributing to BrainPy](https://brainpy.readthedocs.io/en/latest/tutorial_advanced/contributing.html)). 

- [x] model and data parallelization on multiple devices for dense connection models
- [ ] model parallelization on multiple devices for sparse spiking network models
- [ ] data parallelization on multiple devices for sparse spiking network models
- [ ] pipeline parallelization on multiple devices for sparse spiking network models
- [ ] multi-compartment modeling
- [ ] measurements, analysis, and visualization methods for large-scale spiking data
- [ ] Online learning methods for large-scale spiking network models
- [ ] Classical plasticity rules for large-scale spiking network models

