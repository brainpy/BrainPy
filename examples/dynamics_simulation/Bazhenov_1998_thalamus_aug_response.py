# -*- coding: utf-8 -*-

"""
Implementation of the model:

- Bazhenov, Maxim, et al. "Cellular and network models for
  intrathalamic augmenting responses during 10-Hz stimulation."
  Journal of Neurophysiology 79.5 (1998): 2730-2748.
"""

import brainpy as bp


class RE(bp.CondNeuGroup):
  def __init__(self, size):
    super(RE, self).__init__(size, A=1.43e-4)

    self.IL = bp.channels.IL(size, )
    self.IKL = bp.channels.IKL(size, )
    self.INa = bp.channels.INa_TM1991(size, V_sh=-50.)
    self.IK = bp.channels.IK_TM1991(size, V_sh=-50.)
    self.IT = bp.channels.ICaT_HP1992(size, V_sh=0., phi_q=3., phi_p=3.)


class TC(bp.CondNeuGroup):
  def __init__(self, size):
    super(TC, self).__init__(size, A=2.9e-4)

    self.IL = bp.channels.IL(size, )
    self.IKL = bp.channels.IKL(size, )
    self.INa = bp.channels.INa_TM1991(size, V_sh=-50.)
    self.IK = bp.channels.IK_TM1991(size, V_sh=-50.)
    self.IT = bp.channels.ICaT_HM1992(size, V_sh=0., )
    self.IA = bp.channels.IKA1_HM1992(size, V_sh=0., phi_q=3.7255, phi_p=3.7)

    self.Ih = bp.channels.Ih_De1996(size, )
    self.Ca = bp.channels.CalciumFirstOrder(size, )

