import logging

from sympy import *
from sympy.parsing.sympy_parser import parse_expr, convert_xor, auto_number

from experimental.parser_annarchy.Equation import transform_condition
from experimental.parser_annarchy.ParserTemplate import functions_dict, user_functions
from experimental.parser_annarchy.config import _objects


class FunctionParser(object):
    '''
    Class to analyse one equation.
    '''
    def __init__(self, 
                 name, 
                 expression, 
                 description,
                 untouched = [],
                 method='explicit', 
                 type=None):
        '''
        Parameters:

        * name : The name of the variable
        * expression: The expression as a string
        * variables: a list of all the variables in the neuron/synapse
        * local_variables: a list of the local variables
        * global_variables: a list of the global variables
        * method: the numerical method to use for ODEs
        * type: forces the analyser to consider the equation as: simple, cond, ODE, inc
        * untouched: list of terms which should not be modified
        '''

        self.args = description
        self.eq = expression

        # Copy the default functions dictionary
        self.local_dict = functions_dict.copy()
        # Add the arguments to the dictionary
        for arg in self.args:
            self.local_dict[arg] = Symbol(arg)


        # Add custom constants
        for obj in _objects['constants']:
            self.local_dict[obj.name] = Symbol(obj.name)

        # Add other functions    
        self.user_functions = user_functions.copy()
        for func in [func[0] for func in _objects['functions']]:
            self.user_functions[func] = func
            self.local_dict[func] = Function(func)

        # Possibly conditionals (up to 10 per equation... dirty!)
        for i in range(10):
            self.local_dict['__conditional__'+str(i)] = Symbol('__conditional__'+str(i))

    def parse(self, part=None):
        if not part:
            part = self.eq

        expression = transform_condition(part)

        # Check if there is a == in the condition
        if '==' in expression:
            # Is it the only term, or are there other operations?
            if '&' in expression or '|' in expression:
                expression = re.sub(r'([\w\s.]+)==([\w\s.]+)', r'Equality(\1, \2)', expression)
            else:
                terms = expression.split('==')
                expression = 'Equality(' + terms[0] + ', ' + terms[1] + ')'

        # Check if there is a != in the condition
        if '!=' in expression:
            # Is it the only term, or are there other operations?
            if '&' in expression or '|' in expression:
                expression = re.sub(r'([\w\s.]+)!=([\w\s.]+)', r'Not(Equality(\1, \2))', expression)
            else:
                terms = expression.split('!=')
                expression = 'Not(Equality(' + terms[0] + ', ' + terms[1] + '))'

        try:
            eq = parse_expr(expression,
                local_dict = self.local_dict,
                transformations = ((auto_number, convert_xor,))
            )
        except:
            print(expression)
            logging.error('The function depends on unknown variables.')

        return ccode(eq, precision=8,
            user_functions=self.user_functions)

    def dependencies(self):
        "For compatibility with Equation."
        return self.args

