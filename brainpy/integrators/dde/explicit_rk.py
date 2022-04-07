# -*- coding: utf-8 -*-

from typing import Dict
import brainpy.math as bm

from brainpy.integrators.constants import F, DT
from brainpy.integrators.dde.base import DDEIntegrator
from brainpy.integrators.ode import common
from brainpy.integrators.utils import compile_code, check_kws
from brainpy.integrators.dde.generic import register_dde_integrator

__all__ = [
  'ExplicitRKIntegrator',
  'Euler',
  'MidPoint',
  'Heun2',
  'Ralston2',
  'RK2',
  'RK3',
  'Heun3',
  'Ralston3',
  'SSPRK3',
  'RK4',
  'Ralston4',
  'RK4Rule38',
]


class ExplicitRKIntegrator(DDEIntegrator):
  A = []  # The A matrix in the Butcher tableau.
  B = []  # The B vector in the Butcher tableau.
  C = []  # The C vector in the Butcher tableau.

  def __init__(self, f, **kwargs):
    super(ExplicitRKIntegrator, self).__init__(f=f, **kwargs)

    # integrator keywords
    keywords = {
      F: 'the derivative function',
      # DT: 'the precision of numerical integration'
    }
    for v in self.variables:
      keywords[f'{v}_new'] = 'the intermediate value'
      for i in range(1, len(self.A) + 1):
        keywords[f'd{v}_k{i}'] = 'the intermediate value'
      for i in range(2, len(self.A) + 1):
        keywords[f'k{i}_{v}_arg'] = 'the intermediate value'
        keywords[f'k{i}_t_arg'] = 'the intermediate value'
    check_kws(self.arg_names, keywords)

    def integral(*vars, **kwargs):
      pass

    self.build()

  def build(self):
    # step stage
    common.step(self.variables, DT, self.A, self.C, self.code_lines, self.parameters)
    # variable update
    return_args = common.update(self.variables, DT, self.B, self.code_lines)
    # returns
    self.code_lines.append(f'  return {", ".join(return_args)}')
    # compile
    self.integral = compile_code(
      code_scope={k: v for k, v in self.code_scope.items()},
      code_lines=self.code_lines,
      show_code=self.show_code,
      func_name=self.func_name)


class Euler(ExplicitRKIntegrator):
  A = [(), ]
  B = [1]
  C = [0]


register_dde_integrator('euler', Euler)


class MidPoint(ExplicitRKIntegrator):
  A = [(), (0.5,)]
  B = [0, 1]
  C = [0, 0.5]


register_dde_integrator('midpoint', MidPoint)


class Heun2(ExplicitRKIntegrator):
  A = [(), (1,)]
  B = [0.5, 0.5]
  C = [0, 1]


register_dde_integrator('heun2', Heun2)


class Ralston2(ExplicitRKIntegrator):
  A = [(), ('2/3',)]
  B = [0.25, 0.75]
  C = [0, '2/3']


register_dde_integrator('ralston2', Ralston2)


class RK2(ExplicitRKIntegrator):
  def __init__(self, f, beta=2 / 3, var_type=None, dt=None, name=None,
               state_delays: Dict[str, bm.TimeDelay] = None,
               neutral_delays: Dict[str, bm.NeutralDelay] = None):
    self.A = [(), (beta,)]
    self.B = [1 - 1 / (2 * beta), 1 / (2 * beta)]
    self.C = [0, beta]
    super(RK2, self).__init__(f=f, var_type=var_type, dt=dt, name=name,
                              state_delays=state_delays, neutral_delays=neutral_delays)


register_dde_integrator('rk2', RK2)


class RK3(ExplicitRKIntegrator):
  A = [(), (0.5,), (-1, 2)]
  B = ['1/6', '2/3', '1/6']
  C = [0, 0.5, 1]


register_dde_integrator('rk3', RK3)


class Heun3(ExplicitRKIntegrator):
  A = [(), ('1/3',), (0, '2/3')]
  B = [0.25, 0, 0.75]
  C = [0, '1/3', '2/3']


register_dde_integrator('heun3', Heun3)


class Ralston3(ExplicitRKIntegrator):
  A = [(), (0.5,), (0, 0.75)]
  B = ['2/9', '1/3', '4/9']
  C = [0, 0.5, 0.75]


register_dde_integrator('ralston3', Ralston3)


class SSPRK3(ExplicitRKIntegrator):
  A = [(), (1,), (0.25, 0.25)]
  B = ['1/6', '1/6', '2/3']
  C = [0, 1, 0.5]


register_dde_integrator('ssprk3', SSPRK3)


class RK4(ExplicitRKIntegrator):
  A = [(), (0.5,), (0., 0.5), (0., 0., 1)]
  B = ['1/6', '1/3', '1/3', '1/6']
  C = [0, 0.5, 0.5, 1]


register_dde_integrator('rk4', RK4)


class Ralston4(ExplicitRKIntegrator):
  A = [(), (.4,), (.29697761, .15875964), (.21810040, -3.05096516, 3.83286476)]
  B = [.17476028, -.55148066, 1.20553560, .17118478]
  C = [0, .4, .45573725, 1]


register_dde_integrator('ralston4', Ralston4)


class RK4Rule38(ExplicitRKIntegrator):
  A = [(), ('1/3',), ('-1/3', '1'), (1, -1, 1)]
  B = [0.125, 0.375, 0.375, 0.125]
  C = [0, '1/3', '2/3', 1]


register_dde_integrator('rk4_38rule', RK4Rule38)
