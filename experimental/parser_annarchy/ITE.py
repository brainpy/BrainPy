import logging

from experimental.parser_annarchy.Equation import Equation
from experimental.parser_annarchy.Function import FunctionParser


def translate_ITE(name, eq, condition, description, untouched, function=False):
    " Recursively processes the different parts of an ITE statement"

    if function:
        solver = FunctionParser
    else:
        solver = Equation

    def process_condition(condition):
        if_statement = condition[0]
        then_statement = condition[1]
        else_statement = condition[2]

        if_solver = solver(name, if_statement, description,
                          untouched = untouched.keys(),
                          type='cond')
        if_code = if_solver.parse()
        if_deps = if_solver.dependencies()

        if isinstance(then_statement, list): # nested conditional
            then_code, then_deps =  process_condition(then_statement)
        else:
            then_solver = solver(name, then_statement, description,
                          untouched = untouched.keys(),
                          type='return')
            then_code = then_solver.parse().split(';')[0]
            then_deps = then_solver.dependencies()
        
        if isinstance(else_statement, list): # nested conditional
            else_code, else_deps =  process_condition(else_statement)
        else:
            else_solver = solver(name, else_statement, description,
                          untouched = untouched.keys(),
                          type='return')
            else_code = else_solver.parse().split(';')[0]
            else_deps = else_solver.dependencies()

        code = '(' + if_code + ' ? ' + then_code + ' : ' + else_code + ')'
        deps = list(set(if_deps + then_deps + else_deps))
        return code, deps

    # Main equation, where the right part is __conditional__
    translator = solver(name, eq, description,
                          untouched = untouched.keys())
    code = translator.parse()
    deps = translator.dependencies()

    # Process the (possibly multiple) ITE
    for i in range(len(condition)):
        itecode, itedeps =  process_condition(condition[i])
        deps += itedeps

        # Replace
        if isinstance(code, str):
            code = code.replace('__conditional__'+str(i), itecode)
        else:
            code[1] = code[1].replace('__conditional__'+str(i), itecode)

    deps = list(set(deps)) # remove doublings
    return code, deps


def extract_ite(name, eq, description, split=True):
    """ Extracts if-then-else statements and processes them.

    If-then-else statements must be of the form:

    .. code-block:: python

        variable = if condition: ...
                       val1 ...
                   else: ...
                       val2

    Conditional statements can be nested, but they should return only one value!
    """

    def transform(code):
        " Transforms the code into a list of lines."
        res = []
        items = []
        for arg in code.split(':'):
            items.append( arg.strip())
        for i in range(len(items)):
            if items[i].startswith('if '):
                res.append( items[i].strip() )
            elif items[i].endswith('else'):
                res.append(items[i].split('else')[0].strip() )
                res.append('else' )
            else: # the last then
                res.append( items[i].strip() )
        return res


    def parse(lines):
        " Recursive analysis of if-else statements"
        result = []
        while lines:
            if lines[0].startswith('if'):
                block = [lines.pop(0).split('if')[1], parse(lines)]
                if lines[0].startswith('else'):
                    lines.pop(0)
                    block.append(parse(lines))
                result.append(block)
            elif not lines[0].startswith(('else')):
                result.append(lines.pop(0))
            else:
                break
        return result[0]

    # If no if, not a conditional
    if not 'if ' in eq:
        return eq, []

    # Process the equation
    condition = []
    # Eventually split around =
    if split:
        left, right =  eq.split('=', 1)
    else:
        left = ''
        right = eq

    nb_then = len(re.findall(':', right))
    nb_else = len(re.findall('else', right))
    # The equation contains a conditional statement
    if nb_then > 0:
        # A if must be right after the equal sign
        if not right.strip().startswith('if'):
            logging.error(eq, '\nThe right term must directly start with a if statement.')

        # It must have the same number of : and of else
        if not nb_then == 2*nb_else:
            logging.error(eq, '\nConditional statements must use both : and else.')

        multilined = transform(right)
        condition = parse(multilined)
        right = ' __conditional__0 ' # only one conditional allowed in that case
        if split:
            eq = left + '=' + right
        else:
            eq = right
    else:
        print(eq)
        logging.error('Conditional statements must define "then" and "else" values.\n var = if condition: a else: b')

    return eq, [condition]
