# -*- coding: utf-8 -*-
import logging
from experimental.parser_annarchy.AnalyseNeuron import analyse_neuron
from experimental.parser_annarchy.config import _objects
from npbrain.tools import DictPlus


def neurons(parameters="", equations="", spike=None,
            reset=None, refractory=None, functions=None,
            name="", description="", extra_values={}):
    self = DictPlus(_default_names={'rate': "Rate-coded neuron", 'spike': "Spiking neuron"})

    # Store the parameters and equations
    self.parameters = parameters
    self.equations = equations
    self.functions = functions
    self.spike = spike
    self.reset = reset
    self.refractory = refractory
    self.extra_values = extra_values

    # Find the type of the neuron
    self.type = 'spike' if self.spike else 'rate'

    # Reporting
    _objects['neurons'].append(self)
    # if not hasattr(self, '_instantiated'):  # User-defined
    #     pass
    # elif len(self._instantiated) == 0:  # First instantiated of the class
    #     _objects['neurons'].append(self)
    self._rk_neurons_type = len(_objects['neurons'])
    if name:
        self.name = name
    else:
        self.name = self._default_names[self.type]
    if description:
        self.short_description = description
    else:
        if self.type == 'spike':
            self.short_description = "User-defined model of a spiking neuron."
        else:
            self.short_description = "User-defined model of a rate-coded neuron."

    # Analyse the neuron type
    self.description = None
    return self


def synapses(parameters="", equations="", psp=None, operation='sum',
             pre_spike=None, post_spike=None, functions=None,
             pruning=None, creating=None, name=None, description=None,
             extra_values={}):
    self = DictPlus(_default_names={'rate': "Rate-coded synapse", 'spike': "Spiking synapse"})

    # Store the parameters and equations
    self.parameters = parameters
    self.equations = equations
    self.functions = functions
    self.pre_spike = pre_spike
    self.post_spike = post_spike
    self.psp = psp
    self.operation = operation
    self.extra_values = extra_values
    self.pruning = pruning
    self.creating = creating

    # Type of the synapse TODO: smarter
    self.type = 'spike' if pre_spike else 'rate'

    # Check the operation
    if self.type == 'spike' and self.operation != 'sum':
        logging.error('Spiking synapses can only perform a sum of presynaptic potentials.')

    if not self.operation in ['sum', 'min', 'max', 'mean']:
        logging.error('The only operations permitted are: sum (default), min, max, mean.')

    # Description
    self.description = None

    # Reporting
    _objects['synapses'].append(self)
    # if not hasattr(self, '_instantiated'):  # User-defined
    #     pass
    # elif len(self._instantiated) == 0:  # First instantiation of the class
    #     _objects['synapses'].append(self)
    self._rk_synapses_type = len(_objects['synapses'])

    if name:
        self.name = name
    else:
        self.name = self._default_names[self.type]

    if description:
        self.short_description = description
    else:
        if self.type == 'spike':
            self.short_description = "User-defined spiking synapse."
        else:
            self.short_description = "User-defined rate-coded synapse."


if __name__ == '__main__1':
    a = neurons(
        parameters="""
            El = -60.0  : population
            Vr = -60.0  : population
            Erev_exc = 0.0  : population
            Erev_inh = -80.0  : population
            Vt = -50.0   : population
            tau = 20.0   : population
            tau_exc = 5.0   : population
            tau_inh = 10.0  : population
            I = 20.0 : population
        """,
        equations="""
            tau * dv/dt = (El - v) + g_exc * (Erev_exc - v) + g_inh * (Erev_inh - v ) + I
            tau_exc * dg_exc/dt = - g_exc 
            tau_inh * dg_inh/dt = - g_inh 
        """,
        spike="""v > Vt""",
        reset="""v = Vr""",
        refractory=5.0
    )
    description = analyse_neuron(a)
    print(description)


if __name__ == '__main__':
    a = neurons(
        parameters="""
        C = 1.0 # Capacitance
        VL = -59.387 # Leak voltage
        VK = -82.0 # Potassium reversal voltage
        VNa = 45.0 # Sodium reveral voltage
        gK = 36.0 # Maximal Potassium conductance
        gNa = 120.0 # Maximal Sodium conductance
        gL = 0.3 # Leak conductance
        vt = 30.0 # Threshold for spike emission
        I = 0.0 # External current
        """,

        equations="""
        # Previous membrane potential
        prev_V = V

        # Voltage-dependency parameters
        an = 0.01 * (V + 60.0) / (1.0 - exp(-0.1* (V + 60.0) ) )
        am = 0.1 * (V + 45.0) / (1.0 - exp (- 0.1 * ( V + 45.0 )))
        ah = 0.07 * exp(- 0.05 * ( V + 70.0 ))

        bn = 0.125 * exp (- 0.0125 * (V + 70.0))
        bm = 4.0 *  exp (- (V + 70.0) / 80.0)
        bh = 1.0/(1.0 + exp (- 0.1 * ( V + 40.0 )) )

        # Alpha/Beta functions
        dn/dt = an * (1.0 - n) - bn * n : init = 0.3, midpoint
        dm/dt = am * (1.0 - m) - bm * m : init = 0.0, midpoint
        dh/dt = ah * (1.0 - h) - bh * h : init = 0.6, midpoint

        # Membrane equation
        C * dV/dt = gL * (VL - V ) + gK * n**4 * (VK - V) + gNa * m**3 * h * (VNa - V) + I + I * Normal(0.0, 1.0) : midpoint

        """,

        spike="""
        # Spike is emitted when the membrane potential crosses the threshold from below
        (V > vt) and (prev_V <= vt)    
        """,
    )
    description = analyse_neuron(a)
    print(description)
