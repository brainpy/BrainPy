# Change from Version 2.3.1 to Version 2.3.2



This release (under the branch of ``brainpy=2.3.x``) continues to add supports for brain-inspired computation.


## New Features


### 1. New package structure for stable API release

Unstable APIs are all hosted in ``brainpy._src`` module. 
Other APIs are stable, and will be maintained in a long time. 


### 2. New schedulers

- `brainpy.optim.CosineAnnealingWarmRestarts`
- `brainpy.optim.CosineAnnealingLR`
- `brainpy.optim.ExponentialLR`
- `brainpy.optim.MultiStepLR`
- `brainpy.optim.StepLR`


### 3. Others

- support `static_argnums` in `brainpy.math.jit`
- fix bugs of `reset_state()` and `clear_input()` in `brainpy.channels`
- fix jit error checking
