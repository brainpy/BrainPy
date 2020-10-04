import keyword
import logging
import re
import string
from collections.abc import Mapping, Hashable

import sympy
from pyparsing import Combine, Suppress, restOfLine, LineEnd, ParseException
from pyparsing import Group, ZeroOrMore, OneOrMore, Optional, Word, CharsNotIn

from npbrain.parser.sympytools import sympy_to_str
from npbrain.parser.sympytools import str_to_sympy
from npbrain.parser.namespace import DEFAULT_CONSTANTS
from npbrain.parser.namespace import DEFAULT_FUNCTIONS
from npbrain.parser.stringtools import get_identifiers
from npbrain.parser.topsort import topsort

__all__ = ['EquationError',
           'Expression',
           'SingleEquation',
           'Equations',
           'is_constant_over_dt',
           'is_stateful',
           'check_subexpressions',
           'parse_string_equations']


logger = logging.Logger(__name__)

# Equation types (currently simple strings but always use the constants,
# this might get refactored into objects, for example)
PARAMETER = 'parameter'
DIFFERENTIAL_EQUATION = 'differential equation'
SUBEXPRESSION = 'subexpression'

# Definitions of equation structure for parsing with pyparsing
# TODO: Maybe move them somewhere else to not pollute the namespace here?
#       Only IDENTIFIER and EQUATIONS are ever used later
###############################################################################
# Basic Elements
###############################################################################

# identifiers like in C: can start with letter or underscore, then a
# combination of letters, numbers and underscores
# Note that the check_identifiers function later performs more checks, e.g.
# names starting with underscore should only be used internally
IDENTIFIER = Word(string.ascii_letters + '_',
                  string.ascii_letters + string.digits + '_').setResultsName('identifier')

# very broad definition here, expression will be analysed by sympy anyway
# allows for multi-line expressions, where each line can have comments
EXPRESSION = Combine(OneOrMore((CharsNotIn(':#\n') +
                                Suppress(Optional(LineEnd()))).ignore('#' + restOfLine)),
                     joinString=' ').setResultsName('expression')

# a unit
# very broad definition here, again. Whether this corresponds to a valid unit
# string will be checked later
UNIT = Word(string.ascii_letters + string.digits + '*/.- ').setResultsName('unit')

# a single Flag (e.g. "const" or "event-driven")
FLAG = Word(string.ascii_letters, string.ascii_letters + '_- ' + string.digits)

# Flags are comma-separated and enclosed in parantheses: "(flag1, flag2)"
FLAGS = (Suppress('(') + FLAG + ZeroOrMore(Suppress(',') + FLAG) +
         Suppress(')')).setResultsName('flags')

###############################################################################
# Equations
###############################################################################
# Three types of equations
# Parameter:
# x : volt (flags)
PARAMETER_EQ = Group(IDENTIFIER + Suppress(':') + UNIT +
                     Optional(FLAGS)).setResultsName(PARAMETER)

# Static equation:
# x = 2 * y : volt (flags)
STATIC_EQ = Group(IDENTIFIER + Suppress('=') + EXPRESSION + Suppress(':') +
                  UNIT + Optional(FLAGS)).setResultsName(SUBEXPRESSION)

# Differential equation
# dx/dt = -x / tau : volt
DIFF_OP = (Suppress('d') + IDENTIFIER + Suppress('/') + Suppress('dt'))
DIFF_EQ = Group(DIFF_OP + Suppress('=') + EXPRESSION + Suppress(':') + UNIT +
                Optional(FLAGS)).setResultsName(DIFFERENTIAL_EQUATION)

# ignore comments
EQUATION = (PARAMETER_EQ | STATIC_EQ | DIFF_EQ).ignore('#' + restOfLine)
EQUATIONS = ZeroOrMore(EQUATION)


class EquationError(Exception):
    """
    Exception type related to errors in an equation definition.
    """
    pass


class Expression(Hashable):
    """
    Class for representing an expression.

    Parameters
    ----------
    code : str, optional
        The expression. Note that the expression has to be written in a form
        that is parseable by sympy. Alternatively, a sympy expression can be
        provided (in the ``sympy_expression`` argument).
    sympy_expression : sympy expression, optional
        A sympy expression. Alternatively, a plain string expression can be
        provided (in the ``code`` argument).
    """

    def __init__(self, code=None, sympy_expression=None):
        if code is None and sympy_expression is None:
            raise TypeError('Have to provide either a string or a sympy expression')
        if code is not None and sympy_expression is not None:
            raise TypeError('Provide a string expression or a sympy expression, not both')

        if code is None:
            code = sympy_to_str(sympy_expression)
        else:
            # try to convert it to a sympy expression
            str_to_sympy(code)

        self.code = code.strip()
        # : Set of identifiers in the code string
        self.identifiers = get_identifiers(code)

    @property
    def stochastic_variables(self):
        """Stochastic variables in this expression"""
        return set([variable for variable in self.identifiers
                    if variable == 'xi' or variable.startswith('xi_')])

    def split_stochastic(self):
        """
        Split the expression into a stochastic and non-stochastic part.

        Splits the expression into a tuple of one `Expression` objects f (the
        non-stochastic part) and a dictionary mapping stochastic variables
        to `Expression` objects. For example, an expression of the form
        ``f + g * xi_1 + h * xi_2`` would be returned as:
        ``(f, {'xi_1': g, 'xi_2': h})``
        Note that the `Expression` objects for the stochastic parts do not
        include the stochastic variable itself.

        Returns
        -------
        (f, d) : (`Expression`, dict)
            A tuple of an `Expression` object and a dictionary, the first
            expression being the non-stochastic part of the equation and
            the dictionary mapping stochastic variables (``xi`` or starting
            with ``xi_``) to `Expression` objects. If no stochastic variable
            is present in the code string, a tuple ``(self, None)`` will be
            returned with the unchanged `Expression` object.
        """
        stochastic_variables = []
        for identifier in self.identifiers:
            if identifier == 'xi' or identifier.startswith('xi_'):
                stochastic_variables.append(identifier)

        # No stochastic variable
        if not len(stochastic_variables):
            return self, None

        stochastic_symbols = [sympy.Symbol(variable, real=True)
                              for variable in stochastic_variables]

        # Note that collect only works properly if the expression is expanded
        collected = str_to_sympy(self.code).expand().collect(stochastic_symbols, evaluate=False)

        f_expr = None
        stochastic_expressions = {}
        for var, s_expr in collected.items():
            expr = Expression(sympy_expression=s_expr)
            if var == 1:
                if any(s_expr.has(s) for s in stochastic_symbols):
                    raise AssertionError(('Error when separating expression '
                                          '"%s" into stochastic and non-'
                                          'stochastic term: non-stochastic '
                                          'part was determined to be "%s" but '
                                          'contains a stochastic symbol)' % (self.code,
                                                                             s_expr)))
                f_expr = expr
            elif var in stochastic_symbols:
                stochastic_expressions[str(var)] = expr
            else:
                raise ValueError(('Expression "%s" cannot be separated into '
                                  'stochastic and non-stochastic '
                                  'term') % self.code)

        if f_expr is None:
            f_expr = Expression('0.0')

        return f_expr, stochastic_expressions

    def _repr_pretty_(self, p, cycle):
        """
        Pretty printing for ipython.
        """
        if cycle:
            raise AssertionError('Cyclical call of CodeString._repr_pretty')
        # Make use of sympy's pretty printing
        p.pretty(str_to_sympy(self.code))

    def __str__(self):
        return self.code

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.code)

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        return self.code == other.code

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.code)


class SingleEquation(Hashable):
    """
    Class for internal use, encapsulates a single equation or parameter.

    .. note::
        This class should never be used directly, it is only useful as part of
        the `Equations` class.
    
    Parameters
    ----------
    type : {PARAMETER, DIFFERENTIAL_EQUATION, SUBEXPRESSION}
        The type of the equation.
    var_name : str
        The variable that is defined by this equation.
    expr : `Expression`, optional
        The expression defining the variable (or ``None`` for parameters).        
    """

    def __init__(self, type, var_name, expr=None):
        self.type = type
        self.varname = var_name
        self.expr = expr

        # will be set later in the sort_subexpressions method of Equations
        self.update_order = -1

    @property
    def identifiers(self):
        """All identifiers in the RHS of this equation."""
        return self.expr.identifiers if self.expr is not None else set([])

    @property
    def stochastic_variables(self):
        """Stochastic variables in the RHS of this equation"""
        return set([variable for variable in self.identifiers
                    if variable == 'xi' or variable.startswith('xi_')])

    def __str__(self):
        if self.type == DIFFERENTIAL_EQUATION:
            s = 'd' + self.varname + '/dt'
        else:
            s = self.varname

        if self.expr is not None:
            s += ' = ' + str(self.expr)

        return s

    def __repr__(self):
        s = '<' + self.type + ' ' + self.varname

        if self.expr is not None:
            s += ': ' + self.expr.code
        return s

    def __eq__(self, other):
        if not isinstance(other, SingleEquation):
            return NotImplemented
        return self._state_tuple == other._state_tuple

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._state_tuple)


class Equations(Hashable, Mapping):
    """
    Container that stores equations from which models can be created.
    
    String equations can be of any of the following forms:
    
    1. ``dx/dt = f : unit (flags)`` (differential equation)
    2. ``x = f : unit (flags)`` (equation)
    3. ``x : unit (flags)`` (parameter)

    String equations can span several lines and contain Python-style comments
    starting with ``#``    
    
    Parameters
    ----------
    eqs : `str` or list of `SingleEquation` objects
        A multiline string of equations (see above) -- for internal purposes
        also a list of `SingleEquation` objects can be given. This is done for
        example when adding new equations to implement the refractory
        mechanism. Note that in this case the variable names are not checked
        to allow for "internal names", starting with an underscore.
    kwds: keyword arguments
        Keyword arguments can be used to replace variables in the equation
        string. Arguments have to be of the form ``varname=replacement``, where
        `varname` has to correspond to a variable name in the given equation.
        The replacement can be either a string (replacing a name with a new
        name, e.g. ``tau='tau_e'``) or a value (replacing the variable name
        with the value, e.g. ``tau=tau_e`` or ``tau=10*ms``).
    """

    def __init__(self, eqns, **kwds):
        if isinstance(eqns, str):
            self._equations = parse_string_equations(eqns)
            # Do a basic check for the identifiers
            self.check_identifiers()
        else:
            self._equations = {}
            for eq in eqns:
                if not isinstance(eq, SingleEquation):
                    raise TypeError('The list should only contain SingleEquation objects, not %s' % type(eq))
                if eq.varname in self._equations:
                    raise EquationError('Duplicate definition of variable "%s"' % eq.varname)
                self._equations[eq.varname] = eq

        self._equations = self._substitute(kwds)

        # Check for special symbol xi (stochastic term)
        uses_xi = None
        for eq in self._equations.values():
            if eq.expr is not None and 'xi' in eq.expr.identifiers:
                if not eq.type == DIFFERENTIAL_EQUATION:
                    raise EquationError(('The equation defining %s contains the '
                                         'symbol "xi" but is not a differential '
                                         'equation.') % eq.varname)
                elif uses_xi is not None:
                    raise EquationError(('The equation defining %s contains the '
                                         'symbol "xi", but it is already used '
                                         'in the equation defining %s.') %
                                        (eq.varname, uses_xi))
                else:
                    uses_xi = eq.varname

        # rearrange subexpressions
        self._sort_subexpressions()

        #: Cache for equations with the subexpressions substituted
        self._substituted_expressions = None

    def _substitute(self, replacements):
        if len(replacements) == 0:
            return self._equations

        new_equations = {}
        for eq in self.values():
            # Replace the name of a model variable (works only for strings)
            if eq.varname in replacements:
                new_varname = replacements[eq.varname]
                if not isinstance(new_varname, str):
                    raise ValueError(('Cannot replace model variable "%s" '
                                      'with a value') % eq.varname)
                if new_varname in self or new_varname in new_equations:
                    raise EquationError(
                        ('Cannot replace model variable "%s" '
                         'with "%s", duplicate definition '
                         'of "%s".' % (eq.varname, new_varname,
                                       new_varname)))
                # make sure that the replacement is a valid identifier
                Equations.check_identifier(new_varname)
            else:
                new_varname = eq.varname

            if eq.type in [SUBEXPRESSION, DIFFERENTIAL_EQUATION]:
                # Replace values in the RHS of the equation
                new_code = eq.expr.code
                for to_replace, replacement in replacements.items():
                    if to_replace in eq.identifiers:
                        if isinstance(replacement, str):
                            # replace the name with another name
                            new_code = re.sub('\\b' + to_replace + '\\b',
                                              replacement, new_code)
                        else:
                            # replace the name with a value
                            new_code = re.sub('\\b' + to_replace + '\\b',
                                              '(' + repr(replacement) + ')',
                                              new_code)
                        try:
                            Expression(new_code)
                        except ValueError as ex:
                            raise ValueError(
                                ('Replacing "%s" with "%r" failed: %s') %
                                (to_replace, replacement, ex))
                new_equations[new_varname] = SingleEquation(eq.type, new_varname,
                                                            expr=Expression(new_code))
            else:
                new_equations[new_varname] = SingleEquation(eq.type, new_varname)

        return new_equations

    def substitute(self, **kwds):
        return Equations(list(self._substitute(kwds).values()))

    def check_identifiers(self):
        """
        Check all identifiers for conformity with the rules.
        
        Raises
        ------
        ValueError
            If an identifier does not conform to the rules.
        """
        for name in self.names:
            # Check an identifier (usually resulting from an equation string provided by
            # the user) for conformity with the rules. The rules are:
            #
            #    1. Only ASCII characters
            #    2. Starts with a character, then mix of alphanumerical characters and
            #       underscore
            #    3. Is not a reserved keyword of Python
            # -----

            # Check whether the identifier is parsed correctly -- this is always the
            # case, if the identifier results from the parsing of an equation but there
            # might be situations where the identifier is specified directly
            parse_result = list(IDENTIFIER.scanString(name))

            # parse_result[0][0][0] refers to the matched string -- this should be the
            # full identifier, if not it is an illegal identifier like "3foo" which only
            # matched on "foo"
            if len(parse_result) != 1 or parse_result[0][0][0] != name:
                raise SyntaxError('"%s" is not a valid variable name.' % name)

            if keyword.iskeyword(name):
                raise SyntaxError(('"%s" is a Python keyword and cannot be used as a '
                                   'variable.') % name)

            if name.startswith('_'):
                raise SyntaxError(('Variable "%s" starts with an underscore, '
                                   'this is only allowed for variables used '
                                   'internally') % name)

            # Make sure that identifier names do not clash with function names.
            # ----
            if name in DEFAULT_CONSTANTS:
                raise SyntaxError(f'"{name}" is the name of a constant, cannot be used as a '
                                  f'variable name.')

            # Check that an identifier is not using a reserved special variable name. The
            #         special variables are: 't', 'dt', and 'xi', as well as everything starting
            #         with `xi_`.
            # ---
            if (name in ('t', 'dt', 't_in_timesteps', 'xi', 'i', 'N') or name.startswith('xi_')):
                raise SyntaxError(('"%s" has a special meaning in equations and cannot '
                                   'be used as a variable name.') % name)
            # Make sure that identifier names do not clash with function names.
            # ----
            if name in DEFAULT_FUNCTIONS:
                raise SyntaxError('"%s" is the name of a function, cannot be used as a '
                                  'variable name.' % name)

    def get_substituted_expressions(self, variables=None, include_subexpressions=False):
        """
        Return a list of ``(varname, expr)`` tuples, containing all
        differential equations (and optionally subexpressions) with all the
        subexpression variables substituted with the respective expressions.

        Parameters
        ----------
        variables : dict, optional
            A mapping of variable names to `Variable`/`Function` objects.
        include_subexpressions : bool
            Whether also to return substituted subexpressions. Default is ``False``.

        Returns
        -------
        expr_tuples : list of (str, `CodeString`)
            A list of ``(varname, expr)`` tuples, where ``expr`` is a
            `CodeString` object with all subexpression variables substituted
            with the respective expression.
        """
        if self._substituted_expressions is None:
            self._substituted_expressions = []
            substitutions = {}
            for eq in self.ordered:
                # Skip parameters
                if eq.expr is None:
                    continue

                new_sympy_expr = str_to_sympy(eq.expr.code, variables).xreplace(substitutions)
                new_str_expr = sympy_to_str(new_sympy_expr)
                expr = Expression(new_str_expr)

                if eq.type == SUBEXPRESSION:
                    substitutions.update({sympy.Symbol(eq.varname, real=True): str_to_sympy(expr.code, variables)})
                    self._substituted_expressions.append((eq.varname, expr))
                elif eq.type == DIFFERENTIAL_EQUATION:
                    #  a differential equation that we have to check
                    self._substituted_expressions.append((eq.varname, expr))
                else:
                    raise AssertionError('Unknown equation type %s' % eq.type)

        if include_subexpressions:
            return self._substituted_expressions
        else:
            return [(name, expr) for name, expr in self._substituted_expressions
                    if self[name].type == DIFFERENTIAL_EQUATION]

    def _sort_subexpressions(self):
        """
        Sorts the subexpressions in a way that resolves their dependencies
        upon each other. After this method has been run, the subexpressions
        returned by the ``ordered`` property are in the order in which
        they should be updated
        """

        # Get a dictionary of all the dependencies on other subexpressions,
        # i.e. ignore dependencies on parameters and differential equations
        static_deps = {}
        for eq in self._equations.values():
            if eq.type == SUBEXPRESSION:
                exp = [dep for dep in eq.identifiers
                       if dep in self._equations and self._equations[dep].type == SUBEXPRESSION]
                static_deps[eq.varname] = exp

        try:
            sorted_eqs = topsort(static_deps)
        except ValueError:
            raise ValueError('Cannot resolve dependencies between static '
                             'equations, dependencies contain a cycle.')

        # put the equations objects in the correct order
        for order, static_variable in enumerate(sorted_eqs):
            self._equations[static_variable].update_order = order

        # Sort differential equations and parameters after subexpressions
        for eq in self._equations.values():
            if eq.type == DIFFERENTIAL_EQUATION:
                eq.update_order = len(sorted_eqs)
            elif eq.type == PARAMETER:
                eq.update_order = len(sorted_eqs) + 1

    def _get_stochastic_type(self):
        """
        Returns the type of stochastic differential equations (additivive or
        multiplicative). The system is only classified as ``additive`` if *all*
        equations have only additive noise (or no noise).
        
        Returns
        -------
        type : str
            Either ``None`` (no noise variables), ``'additive'`` (factors for
            all noise variables are independent of other state variables or
            time), ``'multiplicative'`` (at least one of the noise factors
            depends on other state variables and/or time).
        """

        if not self.is_stochastic:
            return None

        for _, expr in self.get_substituted_expressions():
            _, stochastic = expr.split_stochastic()
            if stochastic is not None:
                for factor in stochastic.values():
                    if 't' in factor.identifiers:
                        # noise factor depends on time
                        return 'multiplicative'

                    for identifier in factor.identifiers:
                        if identifier in self.diff_eq_names:
                            # factor depends on another state variable
                            return 'multiplicative'

        return 'additive'

    ############################################################################
    # Properties
    ############################################################################

    @property
    def ordered(self):
        """A list of all equations, sorted
       according to the order in which they should
       be updated"""
        return sorted(self._equations.values(),
                      key=lambda key: (key.update_order, key.varname))

    @property
    def diff_eq_expressions(self):
        return [(varname, eq.expr) for varname, eq in self.items()
                if eq.type == DIFFERENTIAL_EQUATION]

    @property
    def eq_expressions(self):
        return [(varname, eq.expr) for varname, eq in self.items()
                if eq.type in (SUBEXPRESSION, DIFFERENTIAL_EQUATION)]

    # Sets of names
    @property
    def names(self):
        """All variable names defined in the equations."""
        return set([eq.varname for eq in self.ordered])

    @property
    def diff_eq_names(self):
        """All differential equation names."""
        return set([eq.varname for eq in self.ordered
                    if eq.type == DIFFERENTIAL_EQUATION])

    @property
    def subexpr_names(self):
        """All subexpression names."""
        return set([eq.varname for eq in self.ordered
                    if eq.type == SUBEXPRESSION])

    @property
    def eq_names(self):
        """All equation names (including subexpressions)."""
        return set([eq.varname for eq in self.ordered
                    if eq.type in (DIFFERENTIAL_EQUATION, SUBEXPRESSION)])

    @property
    def parameter_names(self):
        """All parameter names."""
        return set([eq.varname for eq in self.ordered if eq.type == PARAMETER])

    @property
    def identifiers(self):
        """Set of all identifiers used in the equations, excluding the
        variables defined in the equations"""
        return set().union(*[eq.identifiers for eq in self._equations.values()]) - self.names

    @property
    def stochastic_variables(self):
        return set([variable for variable in self.identifiers
                    if variable == 'xi' or variable.startswith('xi_')])

    @property
    def is_stochastic(self):
        """Whether the equations are stochastic."""
        return len(self.stochastic_variables) > 0

    @property
    def stochastic_type(self):
        return self._get_stochastic_type()

    ############################################################################
    # Representation
    ############################################################################

    def __str__(self):
        strings = [str(eq) for eq in self.ordered]
        return '\n'.join(strings)

    def __repr__(self):
        return '<Equations object consisting of %d equations>' % len(self._equations)

    def __iter__(self):
        return iter(self._equations)

    def __len__(self):
        return len(self._equations)

    def __getitem__(self, key):
        return self._equations[key]

    def __add__(self, other_eqns):
        if isinstance(other_eqns, str):
            other_eqns = parse_string_equations(other_eqns)
        elif not isinstance(other_eqns, Equations):
            return NotImplemented

        return Equations(list(self.values()) + list(other_eqns.values()))

    def __hash__(self):
        return hash(frozenset(self._equations.items()))


def is_constant_over_dt(expression, variables, dt_value):
    """
    Check whether an expression can be considered as constant over a time step.
    This is *not* the case when the expression either:

    1. contains the variable ``t`` (except as the argument of a function that
       can be considered as constant over a time step, e.g. a `TimedArray` with
       a dt equal to or greater than the dt used to evaluate this expression)
    2. refers to a stateful function such as ``rand()``.

    Parameters
    ----------
    expression : `sympy.Expr`
        The (sympy) expression to analyze
    variables : dict
        The variables dictionary.
    dt_value : float or None
        The length of a timestep (without units), can be ``None`` if the
        time step is not yet known.

    Returns
    -------
    is_constant : bool
        Whether the expression can be considered to be constant over a time
        step.
    """
    t_symbol = sympy.Symbol('t', real=True, positive=True)
    if expression == t_symbol:
        return False  # The full expression is simply "t"
    func_name = str(expression.func)
    func_variable = variables.get(func_name, None)
    if func_variable is not None and not func_variable.stateless:
        return False
    for arg in expression.args:
        if arg == t_symbol and dt_value is not None:
            # We found "t" -- if it is not the only argument of a locally
            # constant function we bail out
            if not (func_variable is not None and func_variable.is_locally_constant(dt_value)):
                return False
        else:
            if not is_constant_over_dt(arg, variables, dt_value):
                return False
    return True


def is_stateful(expression, variables):
    """
    Whether the given expression refers to stateful functions (and is therefore
    not guaranteed to give the same result if called repetively).

    Parameters
    ----------
    expression : `sympy.Expression`
        The sympy expression to check.
    variables : dict
        The dictionary mapping variable names to `Variable` or `Function`
        objects.

    Returns
    -------
    stateful : bool
        ``True``, if the given expression refers to a stateful function like
        ``rand()`` and ``False`` otherwise.
    """
    func_name = str(expression.func)
    func_variable = variables.get(func_name, None)
    if func_variable is not None and not func_variable.stateless:
        return True
    for arg in expression.args:
        if is_stateful(arg, variables):
            return True
    return False


def check_subexpressions(group, equations, run_namespace):
    """
    Checks the subexpressions in the equations and raises an error if a
    subexpression refers to stateful functions without being marked as
    "constant over dt".

    Parameters
    ----------
    group : `Group`
        The group providing the context.
    equations : `Equations`
        The equations to check.
    run_namespace : dict
        The run namespace for resolving variables.

    Raises
    ------
    SyntaxError
        For subexpressions not marked as "constant over dt" that refer to
        stateful functions.
    """
    for eq in equations.ordered:
        if eq.type == SUBEXPRESSION:
            # Check whether the expression is stateful (most commonly by
            # referring to rand() or randn()
            variables = group.resolve_all(eq.identifiers,
                                          run_namespace,
                                          # we don't need to raise any warnings
                                          # for the user here, warnings will
                                          # be raised in create_runner_codeobj
                                          user_identifiers=set())
            expression = str_to_sympy(eq.expr.code, variables=variables)

            # Check whether the expression refers to stateful functions
            if is_stateful(expression, variables):
                raise SyntaxError("The subexpression '{}' refers to a stateful "
                                  "function (e.g. rand()). Such expressions "
                                  "should only be evaluated once per timestep, "
                                  "add the 'constant over dt'"
                                  "flag.".format(eq.varname))


def extract_constant_subexpressions(eqs):
    without_const_subexpressions = []
    const_subexpressions = []
    for eq in eqs.ordered:
        if eq.type == SUBEXPRESSION and 'constant over dt' in eq.flags:
            if 'shared' in eq.flags:
                flags = ['shared']
            else:
                flags = None
            without_const_subexpressions.append(
                SingleEquation(PARAMETER, eq.varname))
            const_subexpressions.append(eq)
        else:
            without_const_subexpressions.append(eq)

    return (Equations(without_const_subexpressions),
            Equations(const_subexpressions))


def parse_string_equations(eqns):
    """Parse a string defining equations.

    Parameters
    ----------
    eqns : str
        The (possibly multi-line) string defining the equations. See the
        documentation of the `Equations` class for details.

    Returns
    -------
    equations : dict
        A dictionary mapping variable names to
        `~brian2.equations.equations.Equations` objects
    """
    equations = {}

    try:
        parsed = EQUATIONS.parseString(eqns, parseAll=True)
    except ParseException as p_exc:
        raise EquationError('Parsing failed: \n' + str(p_exc.line) + '\n' +
                            ' ' * (p_exc.column - 1) + '^\n' + str(p_exc))
    for eq in parsed:
        eq_type = eq.getName()
        eq_content = dict(eq.items())
        # Check for reserved keywords
        identifier = eq_content['identifier']

        expression = eq_content.get('expression', None)
        if expression is not None:
            # Replace multiple whitespaces (arising from joining multiline
            # strings) with single space
            p = re.compile(r'\s{2,}')
            expression = Expression(p.sub(' ', expression))

        equation = SingleEquation(eq_type, identifier, expr=expression)

        if identifier in equations:
            raise EquationError('Duplicate definition of variable "%s"' % identifier)

        equations[identifier] = equation

    return equations
