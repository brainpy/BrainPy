
This release provides supports for automatic object-oriented (OO) transformations. 


## What's new?


1. Automatic OO transformations on longer need to take ``dyn_vars`` or ``child_objs`` information.
   These transformations are capable of automatic inferring the underlying dynamical variables. 
   Specifically, they include:
   
   - ``brainpy.math.grad`` and other autograd functionalities
   - ``brainpy.math.jit``
   - ``brainpy.math.for_loop``
   - ``brainpy.math.while_loop``
   - ``brainpy.math.ifelse``
   - ``brainpy.math.cond``

2. Update documentations 


