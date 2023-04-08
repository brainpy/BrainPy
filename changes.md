# Change from Version 2.3.4 to Version 2.3.5


This release continues to add supports for improving the usability of BrainPy.


## New Features


1. New data structures for object-oriented transformations. 
   - ``NodeList`` and ``NodeDict`` for a list/tuple/dict of ``BrainPyObject`` instances.
   - ``ListVar`` and ``DictVar`` for a list/tuple/dict of brainpy data.
2. `Clip` transformation for brainpy initializers.
3. All ``brainpylib`` operators are accessible in ``brainpy.math`` module.
4. Enable monitoring GPU models on CPU when setting ``DSRunner(..., memory_efficient=True)``. This setting can usually reduce so much memory usage.   
5. ``brainpylib`` wheels on the linux platform support the GPU operators. Users can install gpu version of ``brainpylib`` (require ``brainpylib>=0.1.7``) directly by ``pip install brainpylib``.



