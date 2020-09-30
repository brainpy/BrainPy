from experimental.parser_annarchy import CoupledEquations
from experimental.parser_annarchy.Extraction import *


def analyse_synapse(synapse):
    """
    Parses the structure and generates code snippets for the synapse type.

    It returns a ``description`` dictionary with the following fields:

    * 'object': 'synapse' by default, to distinguish it from 'neuron'
    * 'type': either 'rate' or 'spiking'
    * 'raw_parameters': provided field
    * 'raw_equations': provided field
    * 'raw_functions': provided field
    * 'raw_psp': provided field
    * 'raw_pre_spike': provided field
    * 'raw_post_spike': provided field
    * 'parameters': list of parameters defined for the synapse type
    * 'variables': list of variables defined for the synapse type
    * 'functions': list of functions defined for the synapse type
    * 'attributes': list of names of all parameters and variables
    * 'local': list of names of parameters and variables which are local to each synapse
    * 'semiglobal': list of names of parameters and variables which are local to each postsynaptic neuron
    * 'global': list of names of parameters and variables which are global to the projection
    * 'random_distributions': list of random number generators used in the neuron equations
    * 'global_operations': list of global operations (min/max/mean...) used in the equations
    * 'pre_global_operations': list of global operations (min/max/mean...) on the pre-synaptic population
    * 'post_global_operations': list of global operations (min/max/mean...) on the post-synaptic population
    * 'pre_spike': list of variables updated after a pre-spike event
    * 'post_spike': list of variables updated after a post-spike event
    * 'dependencies': dictionary ('pre', 'post') of lists of pre (resp. post) variables accessed by the synapse (used for delaying variables)
    * 'psp': dictionary ('eq' and 'psp') for the psp code to be summed
    * 'pruning' and 'creating': statements for structural plasticity


    Each parameter is a dictionary with the following elements:

    * 'bounds': unused
    * 'ctype': 'type of the parameter: 'float', 'double', 'int' or 'bool'
    * 'eq': original equation in text format
    * 'flags': list of flags provided after the :
    * 'init': initial value
    * 'locality': 'local', 'semiglobal' or 'global'
    * 'name': name of the parameter

    Each variable is a dictionary with the following elements:

    * 'bounds': dictionary of bounds ('init', 'min', 'max') provided after the :
    * 'cpp': C++ code snippet updating the variable
    * 'ctype': type of the variable: 'float', 'double', 'int' or 'bool'
    * 'dependencies': list of variable and parameter names on which the equation depends
    * 'eq': original equation in text format
    * 'flags': list of flags provided after the :
    * 'init': initial value
    * 'locality': 'local', 'semiglobal' or 'global'
    * 'method': numericalmethod for ODEs
    * 'name': name of the variable
    * 'pre_loop': ODEs have a pre_loop term for precomputing dt/tau
    * 'switch': ODEs have a switch term
    * 'transformed_eq': same as eq, except special terms (sums, rds) are replaced with a temporary name
    * 'untouched': dictionary of special terms, with their new name as keys and replacement values as values.

    """

    # Store basic information
    description = {
        'object': 'synapse',
        'type': synapse.type,
        'raw_parameters': synapse.parameters,
        'raw_equations': synapse.equations,
        'raw_functions': synapse.functions
    }

    # Psps is what is actually summed over the incoming weights
    if synapse.psp:
        description['raw_psp'] = synapse.psp
    elif synapse.type == 'rate':
        description['raw_psp'] = "w*pre.r"

    # Spiking synapses additionally store pre_spike and post_spike
    if synapse.type == 'spike':
        description['raw_pre_spike'] = synapse.pre_spike
        description['raw_post_spike'] = synapse.post_spike

    # Extract parameters and variables names
    parameters = extract_parameters(synapse.parameters, synapse.extra_values)
    variables = extract_variables(synapse.equations)

    # Extract functions
    functions = extract_functions(synapse.functions, False)

    # Check the presence of w
    description['plasticity'] = False
    for var in parameters + variables:
        if var['name'] == 'w':
            break
    else:
        parameters.append(
            {
                'name': 'w', 'bounds': {}, 'ctype': config['precision'],
                'init': 0.0, 'flags': [], 'eq': 'w=0.0', 'locality': 'local'
            }
        )

    # Find out a plasticity rule
    for var in variables:
        if var['name'] == 'w':
            description['plasticity'] = True
            break

    # Build lists of all attributes (param+var), which are local or global
    attributes, local_var, global_var, semiglobal_var = get_attributes(parameters, variables, neuron=False)

    # Test if attributes are declared only once
    if len(attributes) != len(list(set(attributes))):
        _error('Attributes must be declared only once.', attributes)


    # Add this info to the description
    description['parameters'] = parameters
    description['variables'] = variables
    description['functions'] = functions
    description['attributes'] = attributes
    description['local'] = local_var
    description['semiglobal'] = semiglobal_var
    description['global'] = global_var
    description['global_operations'] = []

    # Lists of global operations needed at the pre and post populations
    description['pre_global_operations'] = []
    description['post_global_operations'] = []

    # Extract RandomDistribution objects
    description['random_distributions'] = extract_randomdist(description)

    # Extract event-driven info
    if description['type'] == 'spike':
        # pre_spike event
        description['pre_spike'] = extract_pre_spike_variable(description)
        for var in description['pre_spike']:
            if var['name'] in ['g_target']: # Already dealt with
                continue
            for avar in description['variables']:
                if var['name'] == avar['name']:
                    break
            else: # not defined already
                description['variables'].append(
                {'name': var['name'], 'bounds': var['bounds'], 'ctype': var['ctype'], 'init': var['init'],
                 'locality': var['locality'],
                 'flags': [], 'transformed_eq': '', 'eq': '',
                 'cpp': '', 'switch': '', 're_loop': '',
                 'untouched': '', 'method':'explicit'}
                )
                description['local'].append(var['name'])
                description['attributes'].append(var['name'])

        # post_spike event
        description['post_spike'] = extract_post_spike_variable(description)
        for var in description['post_spike']:
            if var['name'] in ['g_target', 'w']: # Already dealt with
                continue
            for avar in description['variables']:
                if var['name'] == avar['name']:
                    break
            else: # not defined already
                description['variables'].append(
                {'name': var['name'], 'bounds': var['bounds'], 'ctype': var['ctype'], 'init': var['init'],
                 'locality': var['locality'],
                 'flags': [], 'transformed_eq': '', 'eq': '',
                 'cpp': '', 'switch': '', 'untouched': '', 'method':'explicit'}
                )
                description['local'].append(var['name'])
                description['attributes'].append(var['name'])

    # Variables names for the parser_annarchy which should be left untouched
    untouched = {}
    description['dependencies'] = {'pre': [], 'post': []}

    # The ODEs may be interdependent (implicit, midpoint), so they need to be passed explicitely to CoupledEquations
    concurrent_odes = []

    # Iterate over all variables
    for variable in description['variables']:
        # Equation
        eq = variable['transformed_eq']
        if eq.strip() == '':
            continue

        # Dependencies must be gathered
        dependencies = []

        # Extract global operations
        eq, untouched_globs, global_ops = extract_globalops_synapse(variable['name'], eq, description)
        description['pre_global_operations'] += global_ops['pre']
        description['post_global_operations'] += global_ops['post']
        # Remove doubled entries
        description['pre_global_operations'] = [i for n, i in enumerate(description['pre_global_operations']) if i not in description['pre_global_operations'][n + 1:]]
        description['post_global_operations'] = [i for n, i in enumerate(description['post_global_operations']) if i not in description['post_global_operations'][n + 1:]]

        # Extract pre- and post_synaptic variables
        eq, untouched_var, prepost_dependencies = extract_prepost(variable['name'], eq, description)

        # Store the pre-post dependencies at the synapse level
        description['dependencies']['pre'] += prepost_dependencies['pre']
        description['dependencies']['post'] += prepost_dependencies['post']
        # and also on the variable for checking
        variable['prepost_dependencies'] = prepost_dependencies

        # Extract if-then-else statements
        eq, condition = extract_ite(variable['name'], eq, description)

        # Add the untouched variables to the global list
        for name, val in untouched_globs.items():
            if not name in untouched.keys():
                untouched[name] = val
        for name, val in untouched_var.items():
            if not name in untouched.keys():
                untouched[name] = val

        # Save the tranformed equation
        variable['transformed_eq'] = eq

        # Find the numerical method if any
        method = find_method(variable)

        # Process the bounds
        if 'min' in variable['bounds'].keys():
            if isinstance(variable['bounds']['min'], str):
                translator = Equation(variable['name'], variable['bounds']['min'],
                                      description,
                                      type = 'return',
                                      untouched = untouched.keys())
                variable['bounds']['min'] = translator.parse().replace(';', '')
                dependencies += translator.dependencies()

        if 'max' in variable['bounds'].keys():
            if isinstance(variable['bounds']['max'], str):
                translator = Equation(variable['name'], variable['bounds']['max'],
                                      description,
                                      type = 'return',
                                      untouched = untouched.keys())
                variable['bounds']['max'] = translator.parse().replace(';', '')
                dependencies += translator.dependencies()

        # Analyse the equation
        if condition == []: # Call Equation
            translator = Equation(variable['name'], eq,
                                  description,
                                  method = method,
                                  untouched = untouched.keys())
            code = translator.parse()
            dependencies += translator.dependencies()

        else: # An if-then-else statement
            code, deps = translate_ITE(variable['name'], eq, condition, description,
                    untouched)
            dependencies += deps

        if isinstance(code, str):
            pre_loop = {}
            cpp_eq = code
            switch = None
        else: # ODE
            pre_loop = code[0]
            cpp_eq = code[1]
            switch = code[2]

        # Replace untouched variables with their original name
        for prev, new in untouched.items():
            cpp_eq = cpp_eq.replace(prev, new)

        # Replace local functions
        for f in description['functions']:
            cpp_eq = re.sub(r'([^\w]*)'+f['name']+'\(', r'\1'+ f['name'] + '(', ' ' + cpp_eq).strip()

        # Store the result
        variable['pre_loop'] = pre_loop # Things to be declared before the for loop (eg. dt)
        variable['cpp'] = cpp_eq # the C++ equation
        variable['switch'] = switch # switch value id ODE
        variable['untouched'] = untouched # may be needed later
        variable['method'] = method # may be needed later
        variable['dependencies'] = dependencies 

        # If the method is implicit or midpoint, the equations must be solved concurrently (depend on v[t+1])
        if method in ['implicit', 'midpoint'] and switch is not None:
            concurrent_odes.append(variable)

    # After all variables are processed, do it again if they are concurrent
    if len(concurrent_odes) > 1 :
        solver = CoupledEquations(description, concurrent_odes)
        new_eqs = solver.parse()
        for idx, variable in enumerate(description['variables']):
            for new_eq in new_eqs:
                if variable['name'] == new_eq['name']:
                    description['variables'][idx] = new_eq

    # Translate the psp code if any
    if 'raw_psp' in description.keys():
        psp = {'eq' : description['raw_psp'].strip() }

        # Extract global operations
        eq, untouched_globs, global_ops = extract_globalops_synapse('psp', " " + psp['eq'] + " ", description)
        description['pre_global_operations'] += global_ops['pre']
        description['post_global_operations'] += global_ops['post']

        # Replace pre- and post_synaptic variables
        eq, untouched, prepost_dependencies = extract_prepost('psp', eq, description)
        description['dependencies']['pre'] += prepost_dependencies['pre']
        description['dependencies']['post'] += prepost_dependencies['post']
        for name, val in untouched_globs.items():
            if not name in untouched.keys():
                untouched[name] = val

        # Extract if-then-else statements
        eq, condition = extract_ite('psp', eq, description, split=False)

        # Analyse the equation
        if condition == []:
            translator = Equation('psp', eq,
                                  description,
                                  method = 'explicit',
                                  untouched = untouched.keys(),
                                  type='return')
            code = translator.parse()
            deps = translator.dependencies()
        else:
            code, deps = translate_ITE('psp', eq, condition, description, untouched)

        # Replace untouched variables with their original name
        for prev, new in untouched.items():
            code = code.replace(prev, new)

        # Store the result
        psp['cpp'] = code
        psp['dependencies'] = deps
        description['psp'] = psp

    # Process event-driven info
    if description['type'] == 'spike':
        for variable in description['pre_spike'] + description['post_spike']:
            # Find plasticity
            if variable['name'] == 'w':
                description['plasticity'] = True

            # Retrieve the equation
            eq = variable['eq']
            
            # Extract if-then-else statements
            eq, condition = extract_ite(variable['name'], eq, description)

            # Extract pre- and post_synaptic variables
            eq, untouched, prepost_dependencies = extract_prepost(variable['name'], eq, description)

            # Update dependencies
            description['dependencies']['pre'] += prepost_dependencies['pre']
            description['dependencies']['post'] += prepost_dependencies['post']
            # and also on the variable for checking
            variable['prepost_dependencies'] = prepost_dependencies

            # Analyse the equation
            dependencies = []
            if condition == []:
                translator = Equation(variable['name'], 
                                      eq,
                                      description,
                                      method = 'explicit',
                                      untouched = untouched)
                code = translator.parse()
                dependencies += translator.dependencies()
            else:
                code, deps = translate_ITE(variable['name'], eq, condition, description, untouched)
                dependencies += deps

            if isinstance(code, list): # an ode in a pre/post statement
                print(eq)
                if variable in description['pre_spike']:
                    logging.error('It is forbidden to use ODEs in a pre_spike term.')
                elif variable in description['posz_spike']:
                    logging.error('It is forbidden to use ODEs in a post_spike term.')
                else:
                    logging.error('It is forbidden to use ODEs here.')

            # Replace untouched variables with their original name
            for prev, new in untouched.items():
                code = code.replace(prev, new)

            # Process the bounds
            if 'min' in variable['bounds'].keys():
                if isinstance(variable['bounds']['min'], str):
                    translator = Equation(
                                    variable['name'],
                                    variable['bounds']['min'],
                                    description,
                                    type = 'return',
                                    untouched = untouched )
                    variable['bounds']['min'] = translator.parse().replace(';', '')
                    dependencies += translator.dependencies()

            if 'max' in variable['bounds'].keys():
                if isinstance(variable['bounds']['max'], str):
                    translator = Equation(
                                    variable['name'],
                                    variable['bounds']['max'],
                                    description,
                                    type = 'return',
                                    untouched = untouched)
                    variable['bounds']['max'] = translator.parse().replace(';', '')
                    dependencies += translator.dependencies()


            # Store the result
            variable['cpp'] = code # the C++ equation
            variable['dependencies'] = dependencies

    # Structural plasticity
    if synapse.pruning:
        description['pruning'] = extract_structural_plasticity(synapse.pruning, description)
    if synapse.creating:
        description['creating'] = extract_structural_plasticity(synapse.creating, description)

    return description
