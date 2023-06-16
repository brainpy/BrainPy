import brainpy as bp
import unittest


class TestNetDefinition(unittest.TestCase):
  def test_define_net1(self):
    E = bp.neurons.LIF(3200, V_rest=-60., V_th=-50., V_reset=-60.,
                       tau=20., tau_ref=5., method='exp_auto',
                       V_initializer=bp.init.Normal(-60., 2.))

    I = bp.neurons.LIF(800, V_rest=-60., V_th=-50., V_reset=-60.,
                       tau=20., tau_ref=5., method='exp_auto',
                       V_initializer=bp.init.Normal(-60., 2.))

    E2E = bp.synapses.Exponential(E, E, bp.conn.FixedProb(prob=0.02), g_max=0.6,
                                  tau=5., output=bp.synouts.COBA(E=0.),
                                  method='exp_auto')

    E2I = bp.synapses.Exponential(E, I, bp.conn.FixedProb(prob=0.02), g_max=0.6,
                                  tau=5., output=bp.synouts.COBA(E=0.),
                                  method='exp_auto')

    I2E = bp.synapses.Exponential(I, E, bp.conn.FixedProb(prob=0.02), g_max=6.7,
                                  tau=10., output=bp.synouts.COBA(E=-80.),
                                  method='exp_auto')

    I2I = bp.synapses.Exponential(I, I, bp.conn.FixedProb(prob=0.02), g_max=6.7,
                                  tau=10., output=bp.synouts.COBA(E=-80.),
                                  method='exp_auto')

    net = bp.Network(E2E, E2I, I2E, I2I, E=E, I=I)

    runner1 = bp.DSRunner(net,
                          monitors=['E.spike', 'I.spike'],
                          inputs=[('E.input', 20.), ('I.input', 20.)])

    runner2 = bp.DSRunner(net,
                          monitors=[('E.spike', E.spike), ('I.spike', I.spike)],
                          inputs=[(E.input, 20.), (I.input, 20.)])

    runner3 = bp.DSRunner(net,
                          monitors=[('E.spike', E.spike), 'I.spike'],
                          inputs=[(E.input, 20.), (I.input, 20.)])

    runner4 = bp.DSRunner(net,
                          monitors={'E.spike': E.spike, 'I.spike': I.spike},
                          inputs=[(E.input, 20.), (I.input, 20.)])

    bp.math.clear_buffer_memory()


