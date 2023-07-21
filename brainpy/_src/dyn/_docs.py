pneu_doc = '''    
    size: int, or sequence of int. The neuronal population size.
    sharding: The sharding strategy. 
    keep_size: bool. Keep the neuron group size.
    mode: Mode. The computing mode.
    name: str. The group name.
'''.strip()

dpneu_doc = '''
    spk_fun: callable. The spike activation function.
    detach_spk: bool.
    method: str. The numerical integration method.
    spk_type: The spike data type.
    spk_reset: The way to reset the membrane potential when the neuron generates spikes.
        This parameter only works when the computing mode is ``TrainingMode``.
        It can be ``soft`` and ``hard``. Default is ``soft``.
'''.strip()

ref_doc = '''
    tau_ref: float, ArrayType, callable. Refractory period length (ms).
    has_ref_var: bool. Whether has the refractory variable. Default is ``False``.
'''.strip()

if_doc = '''
    V_rest: float, ArrayType, callable. Resting membrane potential.
    R: float, ArrayType, callable. Membrane resistance.
    tau: float, ArrayType, callable. Membrane time constant.
    V_initializer: ArrayType, callable. The initializer of membrane potential.
'''.strip()

lif_doc = '''
    V_rest: float, ArrayType, callable. Resting membrane potential.
    V_reset: float, ArrayType, callable. Reset potential after spike.
    V_th: float, ArrayType, callable. Threshold potential of spike.
    R: float, ArrayType, callable. Membrane resistance.
    tau: float, ArrayType, callable. Membrane time constant.
    V_initializer: ArrayType, callable. The initializer of membrane potential.
'''.strip()


ltc_doc = 'with liquid time-constant'

