from sympy import Symbol, Function

from experimental.parser_annarchy.config import _objects

# Dictionary of default elements for the C++ generation
parser_dict = {
      'dt' : Symbol('dt'),
      't' : Symbol('(double(t)*dt)'),
      'w' : Symbol('w%(local_index)s'),
      'g_target': Symbol('sum'), # TODO: still useful?
      't_last': Symbol('((double)(last_spike%(local_index)s)*dt)'),
      't_pre': Symbol('((double)(%(pre_prefix)slast_spike[pre_rank%(local_index)s])*dt)'),
      't_post': Symbol('((double)(%(post_prefix)slast_spike[post_rank%(semiglobal_index)s])*dt)'),
}

# Dictionary of built-in functions for the C++ generation
functions_dict = {
      'pos': Function('positive'),
      'positive': Function('positive'),
      'neg': Function('negative'),
      'negative': Function('negative'),
      'modulo': Function('modulo'),
      'fabs': Function('fabs'),
      'ite': Function('ite', nargs=3),
      'clip': Function('clip'),
      'True': Symbol('true'),
      'False': Symbol('false'),
      'power': Function('power', nargs=2),
}

# Built-in functions with their correct name
user_functions = {
    'pos': 'positive',
    'positive': 'positive',
    'neg': 'negative',
    'negative': 'negative',
    'modulo': 'modulo',
    'fabs': 'fabs',
    'clip': 'clip',
    'ite': 'ite',
    'power': 'power'
}

def create_local_dict(local_attributes, semiglobal_attributes, global_attributes, untouched):
    """
    Creates a dictionary of Sympy symbols based on the attributes of the neuron/synapse.

    *Parameters:*

    * `local_attributes`: local attributes.
    * `semiglobal_attributes`: semiglobal attributes (postsynaptic).
    * `global_attributes`: global attributes.
    * `untouched`: list of variable names which were replaced (e.g. random distributions, targets)

    """

    # Copy the default dictionary of built-in symbols or functions
    local_dict = parser_dict.copy()
    local_dict.update(functions_dict)

    # Add each variable of the neuron depending on its locality
    for var in local_attributes:
        local_dict[var] = Symbol(var + '%(local_index)s')
    for var in semiglobal_attributes:
        local_dict[var] = Symbol(var + '%(semiglobal_index)s')
    for var in global_attributes:
        local_dict[var] = Symbol(var + '%(global_index)s')

    # Add custom constants
    for obj in _objects['constants']:
        if obj.name in local_dict.keys():
            continue
        local_dict[obj.name] = Symbol(obj.name)

    # Add each untouched variable
    for var in untouched:
        local_dict[var] = Symbol(var)

    return local_dict
