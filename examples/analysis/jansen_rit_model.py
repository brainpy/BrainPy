import brainpy as bp
import brainpy.math as bm


class JansenRitModel(bp.DynamicalSystem):
  r"""The Jansen-Rit model, a neural mass model of the dynamic
  interactions between 3 populations:

  - pyramidal cells (PCs)
  - excitatory interneurons (EINs)
  - inhibitory interneurons (IINs)

  Originally, the model has been developed to describe the waxing-and-waning
  of EEG activity in the alpha frequency range (8-12 Hz) in the visual cortex [1].
  In the past years, however, it has been used as a generic model to describe
  the macroscopic electrophysiological activity within a cortical column [2].
  
  By using the linearity of the convolution operation, the dynamic interactions between PCs, EINs and IINs can be
  expressed via 6 coupled ordinary differential equations that are composed of the two operators defined above:
  
  .. math::
  
    \dot V_{pce} &= I_{pce}, \\
    \dot I_{pce} &= \frac{H_e}{\tau_e} c_4 S(c_3 V_{in}) - \frac{2 I_{pce}}{\tau_e} - \frac{V_{pce}}{\tau_e^2}, \\
    \dot V_{pci} &= I_{pci}, \\
    \dot I_{pci} &= \frac{H_i}{\tau_i} c_2 S(c_1 V_{in}) - \frac{2 I_{pci}}{\tau_i} - \frac{V_{pci}}{\tau_i^2}, \\
    \dot V_{in} &= I_{in}, \\
    \dot I_{in} &= \frac{H_e}{\tau_e} S(V_{pce} - V_{pci}) - \frac{2 I_{in}}{\tau_e} - \frac{V_{in}}{\tau_e^2},
  
  where :math:`V_{pce}`, :math:`V_{pci}`, :math:`V_{in}` are used to represent the average membrane potential
  deflection caused by the excitatory synapses at the PC population, the inhibitory synapses at the PC
  population, and the excitatory synapses at both interneuron populations, respectively.
    
  References
  ----------
  .. [1] B.H. Jansen & V.G. Rit (1995) Electroencephalogram and visual evoked
         potential generation in a mathematical model of coupled cortical
         columns. Biological Cybernetics, 73(4): 357-366.
  .. [2] A. Spiegler, S.J. Kiebel, F.M. Atay, T.R. Knösche (2010) Bifurcation analysis of neural
         mass models: Impact of extrinsic inputs and dendritic time constants. NeuroImage, 52(3):
         1041-1058, https://doi.org/10.1016/j.neuroimage.2009.12.081.

  """

  def __init__(self, method='exp_auto'):
    super(JansenRitModel, self).__init__()

    h_e = 3.25e-3
    tau_e = 10e-3
    h_i = 22e-3
    tau_i = 20e-3
    m_max = 5.
    r = 560.
    V_thr = 6e-3
    c = 135.0
    ei_ratio = 4.0
    io_ratio = 3.0
    u = 220.0

    sigmoid = lambda psp: m_max / (1. + bm.exp(r * (V_thr - psp)))

    def d_ein(ein, t, t_ein):
      return t_ein

    def d_t_ein(t_ein, t, ein, pce, pci):
      _a = h_e * c / tau_e * sigmoid(pce - pci)
      _b = (1. / tau_e ** 2.) * ein - (2. / tau_e) * t_ein
      return _a - _b

    def d_iin(iin, t, t_iin):
      return t_iin

    def d_t_iin(t_iin, t, iin, pce, pci):
      return h_e * c / (ei_ratio * tau_e) * sigmoid(pce - pci) - (1. / tau_e) ^ 2. * iin - (
            2. / tau_e) * t_iin

    def d_pce(pce, t, t_pce):
      return t_pce

    def d_t_pce(t_pce, t, pce, ein, Iext):
      _a = h_e / tau_e * (c * 0.8 * sigmoid(ein) + c * Iext / io_ratio + u)
      _b = (1. / tau_e ** 2.) * pce
      _c = (2. / tau_e) * t_pce
      return _a - _b - _c

    def d_pci(pci, t, t_pci):
      return t_pci

    def d_t_pci(t_pci, t, pci, iin):
      _a = h_i * c / (ei_ratio * tau_i) * sigmoid(iin)
      _b = 1. / tau_i ** 2. * pci - (2. / tau_i) * t_pci
      return _a - _b

    self.int_ein = bp.odeint(d_ein, method=method)
    self.int_t_ein = bp.odeint(d_t_ein, method=method)
    self.int_iin = bp.odeint(d_iin, method=method)
    self.int_t_iin = bp.odeint(d_t_iin, method=method)
    self.int_pce = bp.odeint(d_pce, method=method)
    self.int_t_pce = bp.odeint(d_t_pce, method=method)
    self.int_pci = bp.odeint(d_pci, method=method)
    self.int_t_pci = bp.odeint(d_t_pci, method=method)
