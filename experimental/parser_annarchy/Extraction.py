import re

from experimental.parser_annarchy.ITE import *
from experimental.parser_annarchy.Random import available_distributions, distributions_arguments, distributions_equivalents
from experimental.parser_annarchy.config import config
from experimental.parser_annarchy.config import get_constant
from experimental.parser_annarchy.config import list_constants


def extract_randomdist(description):
    " Extracts RandomDistribution objects from all variables"
    rk_rand = 0
    random_objects = []
    for variable in description['variables']:
        # Equation
        eq = variable['eq']
        # Dependencies
        dependencies = []
        # Search for all distributions
        for dist in available_distributions:
            matches = re.findall('(?P<pre>[^\w.])' + dist + '\(([^()]+)\)', eq)
            if matches == ' ':
                continue
            for l, v in matches:

                # Check the arguments
                arguments = v.split(',')

                # Check the number of provided arguments
                if len(arguments) < distributions_arguments[dist]:
                    print(eq)
                    logging.error(
                        'The distribution ' + dist + ' requires ' + str(distributions_arguments[dist]) + 'parameters')
                elif len(arguments) > distributions_arguments[dist]:
                    print(eq)
                    logging.error('Too many parameters provided to the distribution ' + dist)

                # Process the arguments
                processed_arguments = ""
                for idx in range(len(arguments)):
                    try:
                        arg = float(arguments[idx])
                    except:  # A global parameter
                        if arguments[idx].strip() in description['global']:
                            arg = arguments[idx].strip() + "%(global_index)s"
                            dependencies.append(arguments[idx].strip())
                        else:
                            logging.error(arguments[idx] + ' is not a global parameter of '
                                                           'the neuron/synapse. It can not be '
                                                           'used as an argument to the random '
                                                           'distribution ' + dist + '(' + v + ')')

                    processed_arguments += str(arg)
                    if idx != len(arguments) - 1:  # not the last one
                        processed_arguments += ', '

                definition = distributions_equivalents[dist] + '(' + processed_arguments + ')'

                # Store its definition
                desc = {
                    'name': 'rand_' + str(rk_rand),
                    'dist': dist,
                    'definition': definition,
                    'args': processed_arguments,
                    'template': distributions_equivalents[dist],
                    'locality': variable['locality'],
                    'ctype': 'double',
                    'dependencies': dependencies
                }
                rk_rand += 1
                random_objects.append(desc)

                # Replace its definition by its temporary name
                # Problem: when one uses twice the same RD in a single equation (perverse...)
                eq = eq.replace(dist + '(' + v + ')', desc['name'])
                # Add the new variable to the vocabulary
                description['attributes'].append(desc['name'])
                if variable['name'] in description['local']:
                    description['local'].append(desc['name'])
                elif variable['name'] in description['semiglobal']:
                    description['semiglobal'].append(desc['name'])
                else:  # Why not on a population-wide variable?
                    description['global'].append(desc['name'])

        variable['transformed_eq'] = eq

    return random_objects


def extract_globalops_neuron(name, eq, description):
    """ Replaces global operations (mean(r), etc)  with arbitrary names and
    returns a dictionary of changes.
    """
    untouched = {}
    globs = []
    # Global ops
    glop_names = ['min', 'max', 'mean', 'norm1', 'norm2']
    for op in glop_names:
        matches = re.findall('([^\w]*)' + op + '\(([\s\w]*)\)', eq)
        for pre, var in matches:
            if var.strip() in description['local']:
                globs.append({'function': op, 'variable': var.strip()})
                oldname = op + '(' + var + ')'
                newname = '_' + op + '_' + var.strip()
                eq = eq.replace(oldname, newname)
                untouched[newname] = '_' + op + '_' + var.strip()
            else:
                print(eq)
                logging.error('There is no local attribute ' + var + '.')

    return eq, untouched, globs


def extract_globalops_synapse(name, eq, desc):
    """
    Replaces global operations (mean(pre.r), etc)  with arbitrary names and
    returns a dictionary of changes.
    """
    untouched = {}
    globs = {'pre': [],
             'post': []}
    glop_names = ['min', 'max', 'mean', 'norm1', 'norm2']

    for op in glop_names:
        pre_matches = re.findall('([^\w.])' + op + '\(\s*pre\.([\w]+)\s*\)', eq)
        post_matches = re.findall('([^\w.])' + op + '\(\s*post\.([\w]+)\s*\)', eq)

        for pre, var in pre_matches:
            globs['pre'].append({'function': op, 'variable': var.strip()})
            newname = '__pre_' + op + '_' + var.strip()
            eq = re.sub(op + '\(\s*pre\.([\w]+)\s*\)', newname, eq)
            untouched[newname] = '%(pre_prefix)s_' + op + '_' + var

        for pre, var in post_matches:
            globs['post'].append({'function': op, 'variable': var.strip()})
            newname = '__post_' + op + '_' + var.strip()
            eq = re.sub(op + '\(\s*post\.([\w]+)\s*\)', newname, eq)
            untouched[newname] = '%(post_prefix)s_' + op + '_' + var

    return eq, untouched, globs


def extract_prepost(name, eq, description):
    " Replaces pre.var and post.var with arbitrary names and returns a dictionary of changes."

    dependencies = {'pre': [], 'post': []}

    pre_matches = re.findall(r'pre\.([\w]+)', eq)
    post_matches = re.findall(r'post\.([\w]+)', eq)

    untouched = {}
    # Replace all pre.* occurences with a temporary variable
    for var in list(set(pre_matches)):
        if var == 'sum':  # pre.sum(exc)
            def idx_target(val):
                target = val.group(1).strip()
                if target == '':
                    print(eq)
                    logging.error('pre.sum() requires one argument.')

                rep = '_pre_sum_' + target.strip()
                dependencies['pre'].append('sum(' + target + ')')
                untouched[rep] = '%(pre_prefix)s_sum_' + target + '%(pre_index)s'
                return rep

            eq = re.sub(r'pre\.sum\(([\s\w]+)\)', idx_target, eq)
        else:
            dependencies['pre'].append(var)
            eq = re.sub("pre." + var + "([^_\w]+)", "_pre_" + var + "__\g<1>", eq + " ")
            # eq = eq.replace(target, ' _pre_'+var)
            untouched['_pre_' + var + '__'] = '%(pre_prefix)s' + var + '%(pre_index)s'

    # Replace all post.* occurences with a temporary variable
    for var in list(set(post_matches)):
        if var == 'sum':  # post.sum(exc)
            def idx_target(val):
                target = val.group(1).strip()
                if target == '':
                    print(eq)
                    logging.error('post.sum() requires one argument.')

                dependencies['post'].append('sum(' + target + ')')
                rep = '_post_sum_' + target.strip()
                untouched[rep] = '%(post_prefix)s_sum_' + target + '%(post_index)s'
                return rep

            eq = re.sub(r'post\.sum\(([\s\w]+)\)', idx_target, eq)
        else:
            dependencies['post'].append(var)
            eq = re.sub("post." + var + "([^_\w]+)", "_post_" + var + "__\g<1>", eq + " ")
            # eq = eq.replace(target, ' _post_'+var+'__')
            untouched['_post_' + var + '__'] = '%(post_prefix)s' + var + '%(post_index)s'

    return eq, untouched, dependencies


def extract_parameters(description, extra_values={}):
    """ Extracts all variable information from a multiline description."""
    parameters = []
    # Split the multilines into individual lines
    parameter_list = prepare_string(description)

    # Analyse all variables
    for definition in parameter_list:
        # Check if there are flags after the : symbol
        equation, constraint = split_equation(definition)
        # Extract the name of the variable
        name = extract_name(equation)
        if name in ['_undefined', ""]:
            logging.error("Definition can not be analysed: " + equation)

        # Process constraint
        bounds, flags, ctype, init = extract_boundsflags(constraint, equation, extra_values)

        # Determine locality
        for f in ['population', 'postsynaptic', 'projection']:
            if f in flags:
                if f == 'postsynaptic':
                    locality = 'semiglobal'
                else:
                    locality = 'global'
                break
        else:
            locality = 'local'

        # Store the result
        desc = {'name': name,
                'locality': locality,
                'eq': equation,
                'bounds': bounds,
                'flags': flags,
                'ctype': ctype,
                'init': init,
                }
        parameters.append(desc)
    return parameters


def extract_variables(description):
    """ Extracts all variable information from a multiline description."""
    variables = []
    # Split the multilines into individual lines
    variable_list = process_equations(description)
    # Analyse all variables
    for definition in variable_list:
        # Retrieve the name, equation and constraints for the variable
        equation = definition['eq']
        constraint = definition['constraint']
        name = definition['name']
        if name == '_undefined':
            logging.error('The variable', name, 'can not be analysed.')

        # Check the validity of the equation
        check_equation(equation)

        # Process constraint
        bounds, flags, ctype, init = extract_boundsflags(constraint)

        # Determine locality
        for f in ['population', 'postsynaptic', 'projection']:
            if f in flags:
                if f == 'postsynaptic':
                    locality = 'semiglobal'
                else:
                    locality = 'global'
                break
        else:
            locality = 'local'

        # Store the result
        desc = {'name': name,
                'locality': locality,
                'eq': equation,
                'bounds': bounds,
                'flags': flags,
                'ctype': ctype,
                'init': init}
        variables.append(desc)

    return variables


def extract_boundsflags(constraint, equation="", extra_values={}):
    # Process the flags if any
    bounds, flags = extract_flags(constraint)

    # Get the type of the variable (float/int/bool)
    if 'int' in flags:
        ctype = 'int'
    elif 'bool' in flags:
        ctype = 'bool'
    else:
        ctype = config['precision']

    # Get the init value if declared
    if 'init' in bounds.keys():  # Variables: explicitely set in init=xx
        init = bounds['init']
        if ctype == 'bool':
            if init in ['false', 'False', '0']:
                init = False
            elif init in ['true', 'True', '1']:
                init = True
        elif init in list_constants():
            init = get_constant(init)
        elif ctype == 'int':
            init = int(init)
        else:
            init = float(init)

    elif '=' in equation:  # Parameters: the value is in the equation
        init = equation.split('=')[1].strip()

        # Boolean
        if init in ['false', 'False']:
            init = False
            ctype = 'bool'
        elif init in ['true', 'True']:
            init = True
            ctype = 'bool'
        # Constants
        elif init in list_constants():
            init = get_constant(init)
        # Extra-args (obsolete)
        elif init.strip().startswith("'"):
            var = init.replace("'", "")
            init = extra_values[var]
        # Integers
        elif ctype == 'int':
            try:
                init = eval('int(' + init + ')')
            except:
                print(equation)
                logging.error('The value of the parameter is not an integer.')
        # Floats
        else:
            try:
                init = eval('float(' + init + ')')
            except:
                print(equation)
                logging.error('The value of the parameter is not a float.')

    else:  # Default = 0 according to ctype
        if ctype == 'bool':
            init = False
        elif ctype == 'int':
            init = 0
        elif ctype == 'double' or ctype == 'float':
            init = 0.0

    return bounds, flags, ctype, init


def extract_functions(description, local_global=False):
    """ Extracts all functions from a multiline description."""

    if not description:
        return []

    # Split the multilines into individual lines
    function_list = process_equations(description)

    # Process each function
    functions = []
    for f in function_list:
        eq = f['eq']
        var_name, content = eq.split('=', 1)
        # Extract the name of the function
        func_name = var_name.split('(', 1)[0].strip()
        # Extract the arguments
        arguments = (var_name.split('(', 1)[1].split(')')[0]).split(',')
        arguments = [arg.strip() for arg in arguments]

        # Check the function name is not reserved by Sympy
        from inspect import getmembers
        import sympy
        functions_list = [o[0] for o in getmembers(sympy)]
        if func_name in functions_list:
            logging.error('The function name', func_name, 'is reserved by sympy. Use another one.')

        # Extract their types
        types = f['constraint']
        if types == '':
            return_type = config['precision']
            arg_types = [config['precision'] for a in arguments]
        else:
            types = types.split(',')
            return_type = types[0].strip()
            arg_types = [arg.strip() for arg in types[1:]]
        if not len(arg_types) == len(arguments):
            logging.error('You must specify exactly the types of return value and arguments in ' + eq)

        arg_line = ""
        for i in range(len(arguments)):
            arg_line += arg_types[i] + " " + arguments[i]
            if not i == len(arguments) - 1:
                arg_line += ', '

        # Process the content
        eq2, condition = extract_ite('', content,
                                     {'attributes': [], 'local': [], 'global': [], 'variables': [], 'parameters': []},
                                     split=False)
        if condition == []:
            parser = FunctionParser('', content, arguments)
            parsed_content = parser.parse()
        else:
            parsed_content, deps = translate_ITE("", eq2, condition, arguments, {}, function=True)
            arguments = list(set(arguments))  # somehow the entries in arguments are doubled ... ( HD, 23.02.2017 )

        # Create the one-liner
        fdict = {'name': func_name, 'args': arguments, 'content': content, 'return_type': return_type,
                 'arg_types': arg_types, 'parsed_content': parsed_content, 'arg_line': arg_line}
        if not local_global:  # local to a class
            oneliner = """%(return_type)s %(name)s (%(arg_line)s) {return %(parsed_content)s ;};
""" % fdict
        else:  # global
            oneliner = """inline %(return_type)s %(name)s (%(arg_line)s) {return %(parsed_content)s ;};
""" % fdict
        fdict['cpp'] = oneliner
        functions.append(fdict)

    return functions


def get_attributes(parameters, variables, neuron):
    """ Returns a list of all attributes names, plus the lists of
    local/global variables."""
    attributes = []
    local_var = []
    global_var = []
    semiglobal_var = []
    for p in parameters + variables:
        attributes.append(p['name'])
        if neuron:
            if 'population' in p['flags']:
                global_var.append(p['name'])
            elif 'projection' in p['flags']:
                logging.error('The attribute', p['name'], 'belongs to a neuron, the flag "projection" is forbidden.')
            elif 'postsynaptic' in p['flags']:
                logging.error('The attribute', p['name'], 'belongs to a neuron, the flag "postsynaptic" is forbidden.')
            else:
                local_var.append(p['name'])
        else:
            if 'population' in p['flags']:
                logging.error('The attribute', p['name'], 'belongs to a synapse, the flag "population" is forbidden.')
            elif 'projection' in p['flags']:
                global_var.append(p['name'])
            elif 'postsynaptic' in p['flags']:
                semiglobal_var.append(p['name'])
            else:
                local_var.append(p['name'])

    return attributes, local_var, global_var, semiglobal_var


def extract_targets(variables):
    targets = []
    for var in variables:
        # Rate-coded neurons
        code = re.findall('(?P<pre>[^\w.])sum\(\s*([^()]+)\s*\)', var['eq'])
        for l, t in code:
            targets.append(t.strip())
        # Special case for sum()
        if len(re.findall('([^\w.])sum\(\)', var['eq'])) > 0:
            targets.append('__all__')

        # Spiking neurons
        code = re.findall('([^\w.])g_([\w]+)', var['eq'])
        for l, t in code:
            targets.append(t.strip())

    return list(set(targets))


def extract_spike_variable(description):
    cond = prepare_string(description['raw_spike'])
    if len(cond) > 1:
        print(description['raw_spike'])
        logging.error('The spike condition must be a single expression')

    translator = Equation('raw_spike_cond',
                          cond[0].strip(),
                          description)
    raw_spike_code = translator.parse()
    # Also store the variables used in the condition, as it may be needed for CUDA generation
    spike_code_dependencies = translator.dependencies()

    reset_desc = []
    if 'raw_reset' in description.keys() and description['raw_reset']:
        reset_desc = process_equations(description['raw_reset'])
        for var in reset_desc:
            translator = Equation(var['name'], var['eq'],
                                  description)
            var['cpp'] = translator.parse()
            var['dependencies'] = translator.dependencies()

    return {'spike_cond': raw_spike_code,
            'spike_cond_dependencies': spike_code_dependencies,
            'spike_reset': reset_desc}


def extract_pre_spike_variable(description):
    pre_spike_var = []

    # For all variables influenced by a presynaptic spike
    for var in process_equations(description['raw_pre_spike']):
        # Get its name
        name = var['name']
        eq = var['eq']

        # Process the flags if any
        bounds, flags, ctype, init = extract_boundsflags(var['constraint'])

        # Extract if-then-else statements
        # eq, condition = extract_ite(name, raw_eq, description)

        # Append the result of analysis
        pre_spike_var.append({'name': name, 'eq': eq,
                              'locality': 'local',
                              'bounds': bounds,
                              'flags': flags, 'ctype': ctype,
                              'init': init})

    return pre_spike_var


def extract_post_spike_variable(description):
    post_spike_var = []
    if not description['raw_post_spike']:
        return post_spike_var

    for var in process_equations(description['raw_post_spike']):
        # Get its name
        name = var['name']
        eq = var['eq']

        # Process the flags if any
        bounds, flags, ctype, init = extract_boundsflags(var['constraint'])

        # Extract if-then-else statements
        # eq, condition = extract_ite(name, raw_eq, description)

        post_spike_var.append({'name': name, 'eq': eq, 'raw_eq': eq,
                               'locality': 'local',
                               'bounds': bounds, 'flags': flags, 'ctype': ctype, 'init': init})

    return post_spike_var


def extract_stop_condition(pop):
    eq = pop['stop_condition']['eq']
    pop['stop_condition']['type'] = 'any'
    # Check the flags
    split = eq.split(':')
    if len(split) > 1:  # flag given
        eq = split[0]
        flags = split[1].strip()
        split = flags.split(' ')
        for el in split:
            if el.strip() == 'all':
                pop['stop_condition']['type'] = 'all'
    # Convert the expression
    translator = Equation('stop_cond', eq,
                          pop,
                          type='cond')
    code = translator.parse()
    deps = translator.dependencies()

    pop['stop_condition']['cpp'] = '(' + code + ')'
    pop['stop_condition']['dependencies'] = deps


def extract_structural_plasticity(statement, description):
    # Extract flags
    try:
        eq, constraint = statement.rsplit(':', 1)
        bounds, flags = extract_flags(constraint)
    except:
        eq = statement.strip()
        bounds = {}
        flags = []

    # Extract RD
    rd = None
    for dist in available_distributions:
        matches = re.findall('(?P<pre>[^\w.])' + dist + '\(([^()]+)\)', eq)
        for l, v in matches:
            # Check the arguments
            arguments = v.split(',')
            # Check the number of provided arguments
            if len(arguments) < distributions_arguments[dist]:
                print(eq)
                logging.error(
                    'The distribution ' + dist + ' requires ' + str(distributions_arguments[dist]) + 'parameters')
            elif len(arguments) > distributions_arguments[dist]:
                print(eq)
                logging.error('Too many parameters provided to the distribution ' + dist)
            # Process the arguments
            processed_arguments = ""
            for idx in range(len(arguments)):
                try:
                    arg = float(arguments[idx])
                except:  # A global parameter
                    print(eq)
                    logging.error('Random distributions for creating/pruning synapses must use foxed values.')

                processed_arguments += str(arg)
                if idx != len(arguments) - 1:  # not the last one
                    processed_arguments += ', '
            definition = distributions_equivalents[dist] + '(' + processed_arguments + ')'

            # Store its definition
            if rd:
                print(eq)
                logging.error('Only one random distribution per equation is allowed.')

            rd = {'name': 'rand_' + str(0),
                  'origin': dist + '(' + v + ')',
                  'dist': dist,
                  'definition': definition,
                  'args': processed_arguments,
                  'template': distributions_equivalents[dist]}

    if rd:
        eq = eq.replace(rd['origin'], 'rd(rng)')

    # Extract pre/post dependencies
    eq, untouched, dependencies = extract_prepost('test', eq, description)

    # Parse code
    translator = Equation('test', eq,
                          description,
                          method='cond',
                          untouched={})

    code = translator.parse()
    deps = translator.dependencies()

    # Replace untouched variables with their original name
    for prev, new in untouched.items():
        code = code.replace(prev, new)

    # Add new dependencies
    for dep in dependencies['pre']:
        description['dependencies']['pre'].append(dep)
    for dep in dependencies['post']:
        description['dependencies']['post'].append(dep)

    return {'eq': eq, 'cpp': code, 'bounds': bounds, 'flags': flags, 'rd': rd, 'dependencies': deps}


def find_method(variable):
    if 'implicit' in variable['flags']:
        method = 'implicit'
    elif 'semiimplicit' in variable['flags']:
        method = 'semiimplicit'
    elif 'exponential' in variable['flags']:
        method = 'exponential'
    elif 'midpoint' in variable['flags']:
        method = 'midpoint'
    elif 'explicit' in variable['flags']:
        method = 'explicit'
    elif 'exact' in variable['flags']:
        logging.warning(
            'The "exact" flag should now be replaced by "event-driven". It will stop being valid in a future release.')
        method = 'event-driven'
    elif 'event-driven' in variable['flags']:
        method = 'event-driven'
    else:
        method = config['method']

    return method


def check_equation(equation):
    "Makes a formal check on the equation (matching parentheses, etc)"
    # Matching parentheses
    if equation.count('(') != equation.count(')'):
        print(equation)
        logging.error('The number of parentheses does not match.')
