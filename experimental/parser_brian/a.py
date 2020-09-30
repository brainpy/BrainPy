# -*- coding: utf-8 -*-
import re
from pyparsing import Group, ZeroOrMore, OneOrMore, Optional, Word, CharsNotIn
from pyparsing import Combine, Suppress, restOfLine, LineEnd, ParseException, alphas
from pyparsing import nums



# Equation types (currently simple strings but always use the constants,
# this might get refactored into objects, for example)
PARAMETER = 'parameter'
DIFFERENTIAL_EQUATION = 'differential equation'
SUBEXPRESSION = 'subexpression'

# variable types (FLOAT is the only one that is possible for variables that
# have dimensions). These types will be later translated into dtypes, either
# using the default values from the preferences, or explicitly given dtypes in
# the construction of the `NeuronGroup`, `Synapses`, etc. object
FLOAT = 'float'
INTEGER = 'integer'
BOOLEAN = 'boolean'

# Definitions of equation structure for parsing with pyparsing
###############################################################################
# Basic Elements
###############################################################################

# identifiers like in C: can start with letter or underscore, then a
# combination of letters, numbers and underscores
# Note that the check_identifiers function later performs more checks, e.g.
# names starting with underscore should only be used internally
IDENTIFIER = Word(alphas + '_', alphas + nums + '_').setResultsName('identifier')

# very broad definition here, expression will be analysed by sympy anyway
# allows for multi-line expressions, where each line can have comments
EXPRESSION = Combine(
    OneOrMore((CharsNotIn(':#\n') + Suppress(Optional(LineEnd()))).ignore('#' + restOfLine)),
    joinString=' ').setResultsName('expression')

# a unit
# very broad definition here, again. Whether this corresponds to a valid unit
# string will be checked later
UNIT = Word(alphas + nums + '*/.- ').setResultsName('unit')

# a single Flag (e.g. "const" or "event-driven")
FLAG = Word(alphas, alphas + '_- ' + nums)

# Flags are comma-separated and enclosed in parantheses: "(flag1, flag2)"
FLAGS = (Suppress('(') + FLAG + ZeroOrMore(Suppress(',') + FLAG) +
         Suppress(')')).setResultsName('flags')

###############################################################################
# Equations
###############################################################################
# Three types of equations
# Parameter:
# x : volt (flags)
# PARAMETER_EQ = Group(IDENTIFIER + Optional(FLAGS)).setResultsName(PARAMETER)
PARAMETER_EQ = Group(IDENTIFIER).setResultsName(PARAMETER)

# Static equation:
# x = 2 * y : volt (flags)
# STATIC_EQ = Group(IDENTIFIER + Suppress('=') + EXPRESSION + Optional(FLAGS)).setResultsName(SUBEXPRESSION)
STATIC_EQ = Group(IDENTIFIER + Suppress('=') + EXPRESSION).setResultsName(SUBEXPRESSION)

# Differential equation
# dx/dt = -x / tau : volt
DIFF_OP = (Suppress('d') + IDENTIFIER + Suppress('/') + Suppress('dt'))
# DIFF_EQ = Group(DIFF_OP + Suppress('=') + EXPRESSION + Optional(FLAGS)).setResultsName(DIFFERENTIAL_EQUATION)
DIFF_EQ = Group(DIFF_OP + Suppress('=') + EXPRESSION ).setResultsName(DIFFERENTIAL_EQUATION)

# ignore comments
EQUATION = (PARAMETER_EQ | STATIC_EQ | DIFF_EQ).ignore('#' + restOfLine)
EQUATIONS = ZeroOrMore(EQUATION)


# eqs = '''
# dv/dt = (-v + a + 2*sin(2*pi*t/tau))/tau
# a
# '''
# a = EQUATIONS.parseString(eqs, parseAll=True)


eqs = ' dn/dt = an * (1.0 - n) - bn * n'
eqs = 'dn'
a = (STATIC_EQ | DIFF_EQ).parseString(eqs)
# a = PARAMETER_EQ.parseString(eqs)
print(a)

#
# nonlinear_diff_eq = r"d(?P<y>\w+)/dt = (?P<f>[^:]+) (: (?P<dtype>.+))?"
# nonlinear_diff_eq_pattern = re.compile(nonlinear_diff_eq, re.VERBOSE)
#
# eqs = ' dn/dt = an * (1.0 - n) - bn * n'
# a = nonlinear_diff_eq_pattern.match(eqs.replace(' ', ''))
# print(a)
