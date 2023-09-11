``brainpy.math`` module
=======================

.. contents::
   :local:
   :depth: 1

Objects and Variables
---------------------

.. currentmodule:: brainpy.math 
.. automodule:: brainpy.math 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BrainPyObject
   FunAsObject
   Partial
   NodeList
   NodeDict
   node_dict
   node_list
   Variable
   Parameter
   TrainVar
   VariableView
   VarList
   VarDict
   var_list
   var_dict


Object-oriented Transformations
-------------------------------

.. currentmodule:: brainpy.math 
.. automodule:: brainpy.math 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   grad
   vector_grad
   jacobian
   jacrev
   jacfwd
   hessian
   make_loop
   make_while
   make_cond
   cond
   ifelse
   for_loop
   while_loop
   jit
   cls_jit
   to_object
   function


Environment Settings
--------------------

.. currentmodule:: brainpy.math 
.. automodule:: brainpy.math 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   set
   set_environment
   set_float
   get_float
   set_int
   get_int
   set_bool
   get_bool
   set_complex
   get_complex
   set_dt
   get_dt
   set_mode
   get_mode
   enable_x64
   disable_x64
   set_platform
   get_platform
   set_host_device_count
   clear_buffer_memory
   enable_gpu_memory_preallocation
   disable_gpu_memory_preallocation
   ditype
   dftype
   environment
   batching_environment
   training_environment


Array Interoperability
----------------------

.. currentmodule:: brainpy.math 
.. automodule:: brainpy.math 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   as_device_array
   as_jax
   as_ndarray
   as_numpy
   as_variable


Operators for Pre-Syn-Post Conversion
-------------------------------------

.. currentmodule:: brainpy.math 
.. automodule:: brainpy.math 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   pre2post_sum
   pre2post_prod
   pre2post_max
   pre2post_min
   pre2post_mean
   pre2post_event_sum
   pre2post_csr_event_sum
   pre2post_coo_event_sum
   pre2syn
   syn2post_sum
   syn2post
   syn2post_prod
   syn2post_max
   syn2post_min
   syn2post_mean
   syn2post_softmax


Activation Functions
--------------------

.. currentmodule:: brainpy.math 
.. automodule:: brainpy.math 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   celu
   elu
   gelu
   glu
   prelu
   silu
   selu
   relu
   relu6
   rrelu
   hard_silu
   leaky_relu
   hard_tanh
   hard_sigmoid
   tanh_shrink
   hard_swish
   hard_shrink
   soft_sign
   soft_shrink
   softmax
   softmin
   softplus
   swish
   mish
   log_sigmoid
   log_softmax
   one_hot
   normalize
   sigmoid
   identity
   tanh


Delay Variables
---------------

.. currentmodule:: brainpy.math 
.. automodule:: brainpy.math 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   TimeDelay
   LengthDelay
   NeuTimeDelay
   NeuLenDelay
   ROTATE_UPDATE
   CONCAT_UPDATE


Computing Modes
---------------

.. currentmodule:: brainpy.math 
.. automodule:: brainpy.math 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Mode
   NonBatchingMode
   BatchingMode
   TrainingMode
   nonbatching_mode
   batching_mode
   training_mode


``brainpy.math.sparse`` module: Sparse Operators
------------------------------------------------

.. currentmodule:: brainpy.math.sparse 
.. automodule:: brainpy.math.sparse 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   csrmv
   coomv
   seg_matmul
   csr_to_dense
   csr_to_coo
   coo_to_csr


``brainpy.math.event`` module: Event-driven Operators
-----------------------------------------------------

.. currentmodule:: brainpy.math.event 
.. automodule:: brainpy.math.event 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   csrmv
   info


``brainpy.math.jitconn`` module: Just-In-Time Connectivity Operators
--------------------------------------------------------------------

.. currentmodule:: brainpy.math.jitconn 
.. automodule:: brainpy.math.jitconn 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   event_mv_prob_homo
   event_mv_prob_uniform
   event_mv_prob_normal
   mv_prob_homo
   mv_prob_uniform
   mv_prob_normal


``brainpy.math.surrogate`` module: Surrogate Gradient Functions
---------------------------------------------------------------

.. currentmodule:: brainpy.math.surrogate 
.. automodule:: brainpy.math.surrogate 

.. autosummary::
   :toctree: generated/

   Surrogate
   Sigmoid
   sigmoid
   PiecewiseQuadratic
   piecewise_quadratic
   PiecewiseExp
   piecewise_exp
   SoftSign
   soft_sign
   Arctan
   arctan
   NonzeroSignLog
   nonzero_sign_log
   ERF
   erf
   PiecewiseLeakyRelu
   piecewise_leaky_relu
   SquarewaveFourierSeries
   squarewave_fourier_series
   S2NN
   s2nn
   QPseudoSpike
   q_pseudo_spike
   LeakyRelu
   leaky_relu
   LogTailedRelu
   log_tailed_relu
   ReluGrad
   relu_grad
   GaussianGrad
   gaussian_grad
   InvSquareGrad
   inv_square_grad
   MultiGaussianGrad
   multi_gaussian_grad
   SlayerGrad
   slayer_grad
   inv_square_grad2
   relu_grad2



``brainpy.math.random`` module: Random Number Generations
---------------------------------------------------------

.. currentmodule:: brainpy.math.random 
.. automodule:: brainpy.math.random 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   seed
   split_key
   split_keys
   default_rng
   rand
   randint
   random_integers
   randn
   random
   random_sample
   ranf
   sample
   choice
   permutation
   shuffle
   beta
   exponential
   gamma
   gumbel
   laplace
   logistic
   normal
   pareto
   poisson
   standard_cauchy
   standard_exponential
   standard_gamma
   standard_normal
   standard_t
   uniform
   truncated_normal
   bernoulli
   lognormal
   binomial
   chisquare
   dirichlet
   geometric
   f
   hypergeometric
   logseries
   multinomial
   multivariate_normal
   negative_binomial
   noncentral_chisquare
   noncentral_f
   power
   rayleigh
   triangular
   vonmises
   wald
   weibull
   weibull_min
   zipf
   maxwell
   t
   orthogonal
   loggamma
   categorical
   rand_like
   randint_like
   randn_like
   RandomState
   Generator
   DEFAULT


``brainpy.math.linalg`` module: Linear algebra
----------------------------------------------

.. currentmodule:: brainpy.math.linalg 
.. automodule:: brainpy.math.linalg 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   cholesky
   cond
   det
   eig
   eigh
   eigvals
   eigvalsh
   inv
   svd
   lstsq
   matrix_power
   matrix_rank
   norm
   pinv
   qr
   solve
   slogdet
   tensorinv
   tensorsolve
   multi_dot


``brainpy.math.fft`` module: Discrete Fourier Transform
-------------------------------------------------------

.. currentmodule:: brainpy.math.fft 
.. automodule:: brainpy.math.fft 

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   fft
   fft2
   fftfreq
   fftn
   fftshift
   hfft
   ifft
   ifft2
   ifftn
   ifftshift
   ihfft
   irfft
   irfft2
   irfftn
   rfft
   rfft2
   rfftfreq
   rfftn


