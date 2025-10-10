<p align="center">
  	<img alt="Header image of BrainPy - brain dynamics programming in Python." src="https://raw.githubusercontent.com/brainpy/BrainPy/master/images/logo-banner.png" width=80%>
</p> 



<p align="center">
	<a href="https://pypi.org/project/brainpy/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainpy"></a>
	<a href="https://github.com/brainpy/BrainPy"><img alt="LICENSE" src="https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green?style=plastic"></a>
  	<a href="https://brainpy.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation" src="https://readthedocs.org/projects/brainpy/badge/?version=latest"></a>
  	<a href="https://badge.fury.io/py/brainpy"><img alt="PyPI version" src="https://badge.fury.io/py/brainpy.svg"></a>
    <a href="https://github.com/brainpy/BrainPy/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/brainpy/BrainPy/actions/workflows/CI.yml/badge.svg"></a>
    <a href="https://github.com/brainpy/BrainPy/actions/workflows/CI-models.yml"><img alt="Continuous Integration with Models" src="https://github.com/brainpy/BrainPy/actions/workflows/CI-models.yml/badge.svg"></a>
</p>


BrainPy is a flexible, efficient, and extensible framework for computational neuroscience and brain-inspired computation based on the Just-In-Time (JIT) compilation. It provides an integrative ecosystem for brain dynamics programming, including brain dynamics **building**, **simulation**, **training**, **analysis**, etc. 

- **Source**: https://github.com/brainpy/BrainPy
- **Documentation**: https://brainpy.readthedocs.io/
- **Documentation (state-based)**: https://brainpy-state.readthedocs.io/
- **Bug reports**: https://github.com/brainpy/BrainPy/issues
- **Ecosystem**: https://brainmodeling.readthedocs.io/


## Installation

BrainPy is based on Python (>=3.10) and can be installed on Linux (Ubuntu 16.04 or later), macOS (10.12 or later), and Windows platforms. 

```bash
pip install brainpy -U
```

If you want to use BrainPy with different hardware support, please install the corresponding version of BrainPy:

```bash
pip install brainpy[cpu] -U  # install with CPU support only
pip install brainpy[cuda12] -U  # install with CUDA 12.x support
pip install brainpy[cuda13] -U  # install with CUDA 13.x support
pip install brainpy[tpu] -U  # install with TPU support
```


Install the brainpy with the ecosystem packages:

```bash
pip install BrainX -U
```




### Using BrainPy with Binder

We provide a Binder environment for BrainPy. You can use the following button to launch the environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/brainpy/BrainPy-binder/main)




## Ecosystem

- **[BrainPy](https://github.com/brainpy/BrainPy)**: The solution for the general-purpose brain dynamics programming. 
- **[brainpy-examples](https://github.com/brainpy/examples)**: Comprehensive examples of BrainPy computation. 
- **[brain modeling ecosystem](https://brainmodeling.readthedocs.io/)**: A collection of tools and libraries for brain modeling and simulation.
- [《神经计算建模实战》 (Neural Modeling in Action)](https://github.com/c-xy17/NeuralModeling)
- [第一届神经计算建模与编程培训班 (First Training Course on Neural Modeling and Programming)](https://github.com/brainpy/1st-neural-modeling-and-programming-course)
- [第二届神经计算建模与编程培训班 (Second Training Course on Neural Modeling and Programming)](https://github.com/brainpy/2nd-neural-modeling-and-programming-course)



## Citing 

If you are using ``brainpy >= 2.0``, please consider citing the corresponding paper:

```bibtex
@article {10.7554/eLife.86365,
    article_type = {journal},
    title = {BrainPy, a flexible, integrative, efficient, and extensible framework for general-purpose brain dynamics programming},
    author = {Wang, Chaoming and Zhang, Tianqiu and Chen, Xiaoyu and He, Sichao and Li, Shangyang and Wu, Si},
    editor = {Stimberg, Marcel},
    volume = 12,
    year = 2023,
    month = {dec},
    pub_date = {2023-12-22},
    pages = {e86365},
    citation = {eLife 2023;12:e86365},
    doi = {10.7554/eLife.86365},
    url = {https://doi.org/10.7554/eLife.86365},
    abstract = {Elucidating the intricate neural mechanisms underlying brain functions requires integrative brain dynamics modeling. To facilitate this process, it is crucial to develop a general-purpose programming framework that allows users to freely define neural models across multiple scales, efficiently simulate, train, and analyze model dynamics, and conveniently incorporate new modeling approaches. In response to this need, we present BrainPy. BrainPy leverages the advanced just-in-time (JIT) compilation capabilities of JAX and XLA to provide a powerful infrastructure tailored for brain dynamics programming. It offers an integrated platform for building, simulating, training, and analyzing brain dynamics models. Models defined in BrainPy can be JIT compiled into binary instructions for various devices, including Central Processing Unit (CPU), Graphics Processing Unit (GPU), and Tensor Processing Unit (TPU), which ensures high running performance comparable to native C or CUDA. Additionally, BrainPy features an extensible architecture that allows for easy expansion of new infrastructure, utilities, and machine-learning approaches. This flexibility enables researchers to incorporate cutting-edge techniques and adapt the framework to their specific needs},
    journal = {eLife},
    issn = {2050-084X},
    publisher = {eLife Sciences Publications, Ltd},
}
```


If you want to cite ``brainpy 1.0``, please consider using the corresponding paper:

```bibtex
@inproceedings{wang2021just,
    title={A Just-In-Time Compilation Approach for Neural Dynamics Simulation},
    author={Wang, Chaoming and Jiang, Yingqian and Liu, Xinyu and Lin, Xiaohan and Zou, Xiaolong and Ji, Zilong and Wu, Si},
    booktitle={International Conference on Neural Information Processing},
    pages={15--26},
    year={2021},
    organization={Springer}
}
```


