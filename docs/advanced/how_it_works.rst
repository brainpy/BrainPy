How BrainPy works
=================

The goal of ``BrainPy`` is to provide a highly flexible and
efficient neural simulator for Python users. Specifically,
several principles are kept in mind during the development of BrainPy.

- **Easy to learn and use**.
  The aim of BrainPy is to accelerate your reaches on neuronal dynamics modeling.
  We don't want BrainPy make you pay much time on the learning of how to code. 
  On the contrary, all you need is to focus on the implementation logic
  of the network model by using your familiar `NumPy`_
  APIs. Although you've never used NumPy, even you are unfamiliar with Python, using
  BrainPy is also a easy thing. This is because the
  `Python`_ and `NumPy`_ syntax is simple, elegant and human-like.

- **Flexible and Transparent**.
  Another consideration of BrainPy is the flexibility. 
  Traditional simulators with `code generation`_ approach (such as Brain2 and ANNarchy) 
  have intrinsic limitations. In order to generate efficient low-level (such as c++) codes, 
  these simulators make assumptions for models to simulate, and require users to 
  provide string descriptions to define models. Such string descriptions greatly reduce
  the programming capacity. Moreover, there will always be exceptions beyond the 
  framework assumptions, such as the data or logical flows that the framework do not 
  consider before. Once such frameworks are not tailored to the user needs, extensions 
  becomes difficult and even impossible. Furthermore, no framework is immune to errors when dealing with
  user's incredible models (even the well-tested framework `TensorFlow`_). Therefore, making the
  framework transparent to users becomes indispensable. Considering this, 
  BrainPy enables the users to directly modify the final formatted code once some errors are 
  found (see examples comming soon). 
  Actually, BrainPy endows the users with the fully data/logic flow control.  
  It is concise and powerful, and there is no secrets for users.
  
- **Efficient**.
  The final consideration of BrainPy is to accelerate the running speed of
  of your coded models. In order to achieve high efficiency, we incorporate several 
  Just-In-Time compilers (now support Numba, future will support JAX and others) 
  into BrainPy. Moreover, an unified NumPy-like API are provided for these compilers. 
  The aim of the API design is to let the user *code once and run everywhere*
  (the same code runs on CPU, multi-core, GPU, OpenCL, etc.).


Continue ...

.. _code generation: https://www.frontiersin.org/articles/10.3389/fninf.2018.00068/full
.. _Python: https://www.w3schools.com/python/
.. _NumPy: https://numpy.org/doc/stable/
.. _TensorFlow: https://www.reddit.com/r/MachineLearning/comments/hrawam/d_theres_a_flawbug_in_tensorflow_thats_preventing/
