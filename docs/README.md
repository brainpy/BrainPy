# BrainPy Version 2 Documentation

This directory contains documentation for BrainPy 2.x, the previous major version of BrainPy.

## Overview

BrainPy 2.x is a highly flexible and extensible framework targeting general-purpose Brain Dynamics Programming (BDP). This documentation is maintained for users who are still using BrainPy 2.x.

## Important Note

**As of September 2025, BrainPy has been upgraded to version 3.x.** If you are using BrainPy 3.x, please refer to the main documentation.

To use BrainPy 2.x APIs within version 3.x installations, update your imports:

```python
# Old version (v2.x standalone)
import brainpy as bp
import brainpy.math as bm

# Using v2.x API in BrainPy 3.x
import brainpy as bp
import brainpy.math as bm
```

## Documentation Contents

- **index.rst** - Main documentation entry point
- **core_concepts.rst** - Fundamental concepts of Brain Dynamics Programming
- **tutorials.rst** - Step-by-step tutorials
- **advanced_tutorials.rst** - Advanced usage patterns
- **toolboxes.rst** - Specialized toolboxes for different applications
- **api.rst** - Complete API reference
- **FAQ.rst** - Frequently asked questions
- **brainpy-changelog.md** - Version history and changes
- **brainpylib-changelog.md** - BrainPyLib backend changes

## Building Documentation

This documentation is written in reStructuredText (RST) format and can be built using Sphinx:

```bash
cd docs_classic
make html  # or use the appropriate build script
```

## Installation

For BrainPy 2.x compatibility:

```bash
pip install -U brainpy[cpu]
```

## Learn More

- [Core Concepts](core_concepts.rst) - Understand the fundamentals
- [Tutorials](tutorials.rst) - Learn through examples
- [API Documentation](api.rst) - Complete reference
- [BrainPy Examples](https://brainpy-v2.readthedocs.io/projects/examples/) - Code examples
- [BrainPy Ecosystem](https://brainmodeling.readthedocs.io) - Related projects

## Support

For questions and support, please visit the [BrainPy GitHub repository](https://github.com/brainpy/BrainPy).
