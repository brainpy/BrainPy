# Changelog

## Version 3.0.1

**Release Date:** October 2025

This is a patch release focusing on documentation improvements and module structure cleanup following the 3.0.0 release.

### Major Changes

#### Module Renaming
- **BREAKING CHANGE**: Renamed `brainpy.state_based` module to `brainpy.state`
  - All functionality previously in `brainpy.state_based` is now accessible via `brainpy.state`
  - Users should update imports from `brainpy.state_based` to `brainpy.state`
  - This change provides a cleaner, more intuitive API structure

#### Code Structure Cleanup
- **Removed `brainpy.version2` module**: All BrainPy 2.x functionality has been consolidated
  - The `version2` namespace has been removed from the codebase
  - All version2 functionality is now directly accessible through the main `brainpy` module
  - Version-specific imports are no longer needed

### Documentation

#### Documentation Reorganization
- Renamed `docs_version2` to `docs_classic` for BrainPy 2.x documentation
- Renamed `docs_state_based` to `docs_state` for BrainPy 3.x documentation
- Renamed `examples_version2` to `examples_classic` for consistency
- Renamed `examples_state_based` to `examples_state` for clarity

#### Documentation Updates
- Updated all documentation references to use `brainpy.state` instead of `brainpy.state_based` (#791, #790)
- Updated API documentation structure for improved clarity
- Simplified API reference pages by removing redundant content
- Updated card links and descriptions for `brainpy.state` APIs
- Improved quickstart tutorial (5min-tutorial.ipynb) with clearer examples
- Updated core concepts documentation to reflect new module structure
- Enhanced tutorials with corrected module references
- Updated all example files to use new module structure

#### Examples Updates
- Updated simulation examples (EI networks, COBA, CUBA models) to use new API
- Updated training examples (surrogate gradient training, MNIST models) with correct imports
- Updated gamma oscillation examples with proper module references

### Bug Fixes

#### Testing
- Removed redundant test for abstract Neuron class that was causing conflicts (d06bb47f)

### Migration Guide

For users upgrading from BrainPy 3.0.0:

1. **Update module imports**: Replace `brainpy.state_based` with `brainpy.state`
   ```python

   # New code (BrainPy 3.0.1)
   from brainpy.state import LIF, Expon
   ```

2. **Remove version2 references**: If you were using `brainpy.version2`, migrate to the main `brainpy` module
   ```python
   # Old code (not recommended)
   import brainpy.version2 as bp

   # New code
   import brainpy as bp
   ```

3. **Update documentation references**: If you're linking to documentation, use the new paths:
   - Classic docs: `docs_classic/` (formerly `docs_version2/`)
   - State-based docs: `docs_state/` (formerly `docs_state_based/`)

### Notes
- This release maintains full backward compatibility with BrainPy 3.0.0 except for the module naming changes
- The `brainpy.state_based` to `brainpy.state` rename provides a cleaner API and better reflects the module's purpose
- Documentation is now better organized with clear separation between classic (2.x) and state-based (3.x) APIs




## Version 3.0.0

**Release Date:** October 2025

This is a major release with significant architectural changes and improvements. BrainPy 3.0.0 introduces a new API design while maintaining backward compatibility through the `brainpy` module.

### Major Changes

#### Architecture Reorganization
- **BREAKING CHANGE**: All existing BrainPy 2.x functionality has been moved to `brainpy` module
  - Users can migrate existing code by replacing `import brainpy` with `import brainpy as brainpy`
  - The old `brainpy._src` module structure has been completely reorganized into `brainpy`
  - All submodules (math, dyn, dnn, etc.) are now under `brainpy.*`

#### New Core API (brainpy.*)
- Introduced simplified, streamlined API in the main `brainpy` namespace
- New core modules include:
  - Base classes for neurons and synapses
  - LIF (Leaky Integrate-and-Fire) neuron models
  - Exponential synapse models
  - Synaptic projection modules
  - Short-term plasticity (STP) models
  - Input current generators
  - Readout layers
  - Error handling utilities

### Dependencies
- **Updated**: `brainstate>=0.2.0` (was `>=0.1.0`)
- **Updated**: `brainevent>=0.0.4` (new requirement)
- **Updated**: `braintools>=0.0.9` (integrated into brainpy)
- **Removed**: Hard dependency on `taichi` and `numba` - now optional
- **Updated**: JAX compatibility improvements for version 0.5.0+

### Features

#### Integration of Brain Ecosystem Libraries
- Integrated `brainstate` for state management (#763)
- Integrated `brainevent` for event-driven computations (#771)
- Integrated `braintools` utilities and formatting (#769)

#### Math Module Enhancements (version2.math)
- Added event-driven sparse matrix @ matrix operators (#613)
- Added `ein_rearrange`, `ein_reduce`, and `ein_repeat` functions (#590)
- Added `unflatten` function and `Unflatten` layer (#588)
- Added JIT weight matrix methods (Uniform & Normal) for `dnn.linear` (#673)
- Added JIT connect matrix method for `dnn.linear` (#672)
- Replaced math operators with `braintaichi` for better performance (#698)
- Support for custom operators using CuPy (#653)
- Taichi operators as default customized operators (#598)
- Enhanced taichi custom operator support with GPU backend (#655)
- Support for more than 8 parameters in taichi GPU operator customization (#642)
- Rebased operator customization using MLIR registration interface (#618)
- Added transparent taichi caches with clean caches function (#596)
- Support for taichi customized op with metal CPU backend (#579)
- Improved variable retrieval system (#589)

#### Deep Learning (version2.dnn)
- Improved error handling in `dnn/linear` module (#704)
- Enhanced activation functions and layers

#### Dynamics (version2.dyn)
- Refactored STDP weight update logic requiring `brainevent>=0.0.4` (#771)
- Fixed STDP and training workflows for JAX compatibility (#772)
- Enhanced dual exponential synapse model with `normalize` parameter
- Improved alpha synapse implementation
- Added `clear_input` in the `step_run` function (#601)

#### Integrators (version2.integrators)
- Support for `Integrator.to_math_expr()` (#674)
- Fixed dtype checking during exponential Euler method
- Added `disable_jit` support in `brainpy.math.scan` (#606)
- Fixed `brainpy.math.scan` implementation (#604)

#### Optimizers (version2.optim)
- Fixed AdamW optimizer initialization where "amsgrad" was used before being defined (#660)

#### Tools & Utilities (version2.tools)
- Added `brainpy.tools.compose` and `brainpy.tools.pipe` functions (#624)

### Bug Fixes

#### JAX Compatibility
- Updated JAX import paths for compatibility with version 0.5.0+ (#722)
- Fixed compatibility issues with latest JAX versions (#691, #708, #716)
- Replaced `jax.experimental.host_callback` with `jax.pure_callback` (#670)
- Fixed `test_ndarray.py` for latest JAX version (#708)

#### Math & Operations
- Fixed `CustomOpByNumba` with `multiple_results=True` (#671)
- Updated `CustomOpByNumba` to support JAX version >= 0.4.24 (#669)
- Fixed `brainpy.math.softplus` and `brainpy.dnn.SoftPlus` (#581)
- Fixed bugs in `truncated_normal` and added `TruncatedNormal` initialization (#583, #584, #585, #574, #575)
- Fixed autograd functionality (#687)
- Fixed order of return values in `__load_state__` (#749)

#### Delay & Timing
- Fixed delay bugs including DelayVar in concat mode (#632, #650)
- Fixed wrong randomness in OU process input (#715)

#### UI & Progress
- Fixed progress bar display and update issues (#683)
- Fixed incorrect verbose of `clear_name_cache()` (#681)

#### Python Compatibility
- Replaced `collections.Iterable` with `collections.abc.Iterable` for Python 3.10+ (#677)
- Fixed surrogate gradient function for numpy 2.0 compatibility (#679)

#### Interoperability
- Fixed Flax RNN interoperation (#665)
- Fixed issue with external library integration (#661, #662)

#### Exception Handling
- Fixed exception handling for missing braintaichi module in dependency check (#746)

### Testing & CI

#### Python Support
- Added CI support for Python 3.12 (#705)
- Added CI support for Python 3.13
- Updated supported Python versions: 3.10, 3.11, 3.12, 3.13

#### CI Improvements
- Updated GitHub Actions:
  - `actions/setup-python` from 5 to 6 (#783)
  - `actions/checkout` from 4 to 5 (#773)
  - `actions/first-interaction` from 1 to 3 (#782)
  - `actions/labeler` from 5 to 6 (#781)
  - `actions/download-artifact` from 4 to 5 (#780)
  - `actions/stale` from 9 to 10 (#779)
  - `docker/build-push-action` from 5 to 6 (#678)
- Added greetings workflow and labeler configuration
- Enhanced issue templates and CI configurations

### Documentation

#### Major Documentation Overhaul
- Introduced new BrainPy 3.0 documentation and tutorials (#787)
- Added comprehensive documentation and examples for BrainPy 3.x (#785)
- Updated documentation links for BrainPy 3.0 and 2.0 (#786)
- Implemented dynamic configuration loading for Read the Docs (#784)
- Added Colab and Kaggle links for documentation notebooks (#614, #619)
- Added Chinese version of `operator_custom_with_cupy.ipynb` (#659)
- Fixed various documentation build issues and path references

#### Citation & Acknowledgments
- Added BrainPy citation information (#770)
- Updated ACKNOWLEDGMENTS.md

#### Installation
- Refined installation instructions (#767)
- Updated docstring and parameter formatting (#766)
- Updated README with ecosystem information

### Performance & Memory Management
- Enabled `clear_buffer_memory()` to support clearing `array`, `compilation`, and `names` (#639)
- Cleaned taichi AOT caches and enabled `numpy_func_return` setting (#643)
- Made taichi caches more transparent (#596)
- Enabled BrainPy objects as pytree for direct use with `jax.jit` (#625)

### Object-Oriented Transformations
- Standardized and generalized object-oriented transformations (#628)

### Development & Contributing
- Updated CONTRIBUTING.md with new guidelines
- Added CODEOWNERS file
- Updated SECURITY.md
- License updated to Apache License 2.0

### Removed
- Removed Docker workflow
- Removed hard dependencies on `taichi` and `numba` (#635)
- Removed op register functionality (#700)
- Removed deprecated deprecation files and old module structure
- Removed unnecessary dependencies (#703)

### Migration Guide

For users upgrading from BrainPy 2.6.x:

1. **Keep using BrainPy 2.x API**: Replace imports with `brainpy`
   ```python
   # Old code (BrainPy 2.x)
   import brainpy as bp

   # New code (BrainPy 3.0 with backward compatibility)
   import brainpy as bp
   ```

2. **Adopt new BrainPy 3.0 API**: Explore the simplified API in the main `brainpy` namespace for new projects

3. **Update dependencies**: Ensure `brainstate>=0.2.0`, `brainevent>=0.0.4`, and `braintools>=0.0.9` are installed

4. **Review breaking changes**: Check if your code uses any of the reorganized internal modules

### Notes
- This release maintains backward compatibility through `brainpy`
- The new API in the main `brainpy` namespace represents the future direction of the library
- Documentation for both versions is available on Read the Docs



