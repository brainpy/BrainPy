# -*- coding: utf-8 -*-

"""
The equation parsing in this module are taken example from ``ANNarchy`` and ``dana``.
"""
import operator
import re
from abc import abstractmethod, ABCMeta
from collections.abc import Hashable
from functools import reduce

import sympy
from pyparsing import CharsNotIn
from pyparsing import Combine
from pyparsing import Group
from pyparsing import LineEnd
from pyparsing import Literal
from pyparsing import OneOrMore
from pyparsing import Optional
from pyparsing import ParseException
from pyparsing import Suppress
from pyparsing import Word
from pyparsing import ZeroOrMore
from pyparsing import alphas
from pyparsing import nums
from pyparsing import restOfLine
from sympy.core.sympify import SympifyError

from experimental.parser_brian.sympytools import DEFAULT_CONSTANTS
from experimental.parser_brian.sympytools import DEFAULT_FUNCTIONS
from experimental.parser_brian.sympytools import str_to_sympy
from experimental.parser_brian.sympytools import sympy_to_str

__all__ = ['Expression', 'Equation', 'Equations']

KEYWORDS = {'and', 'or', 'not', 'True', 'False'}

# Equation types (currently simple strings but always use the
# constants, this might get refactored into objects, for example)
PARAMETER = 'parameter'
DIFFERENTIAL_EQUATION = 'differential equation'
SUBEXPRESSION = 'subexpression'

# identifiers like in C: can start with letter or underscore, then a
# combination of letters, numbers and underscores
# Note that the check_identifiers function later performs more checks, e.g.
# names starting with underscore should only be used internally
P_IDENTIFIER = Word(alphas + '_', alphas + nums + '_').setResultsName('identifier')

# very broad definition here, expression will be analysed by sympy anyway
# allows for multi-line expressions, where each line can have comments
P_EXPRESSION = Combine(
    OneOrMore((CharsNotIn(':#\n') + Suppress(Optional(LineEnd()))).ignore('#' + restOfLine)),
    joinString=' ').setResultsName('expression')

# Parameter:
# x
P_PARAMETER_EQ = Group(P_IDENTIFIER).setResultsName(PARAMETER)

# Static equation:
# x = 2 * y
P_STATIC_EQ = Group(P_IDENTIFIER + Suppress('=') + P_EXPRESSION).setResultsName(SUBEXPRESSION)

# Differential equation
# dx/dt = -x / tau
P_DIFF_OP = (Suppress('d') + P_IDENTIFIER + Suppress('/') + Suppress('dt'))
P_DIFF_EQ = Group(P_DIFF_OP + Suppress('=') + P_EXPRESSION).setResultsName(DIFFERENTIAL_EQUATION)


def get_identifiers(expr, include_numbers=False):
    """
    Return all the identifiers in a given string ``expr``, that is everything
    that matches a programming language variable like expression, which is
    here implemented as the regexp ``\\b[A-Za-z_][A-Za-z0-9_]*\\b``.

    Parameters
    ----------
    expr : str
        The string to analyze
    include_numbers : bool, optional
        Whether to include number literals in the output. Defaults to ``False``.

    Returns
    -------
    identifiers : set
        A set of all the identifiers (and, optionally, numbers) in `expr`.

    Examples
    --------
    >>> expr = '3-a*_b+c5+8+f(A - .3e-10, tau_2)*17'
    >>> ids = get_identifiers(expr)
    >>> print(sorted(list(ids)))
    ['A', '_b', 'a', 'c5', 'f', 'tau_2']
    >>> ids = get_identifiers(expr, include_numbers=True)
    >>> print(sorted(list(ids)))
    ['.3e-10', '17', '3', '8', 'A', '_b', 'a', 'c5', 'f', 'tau_2']
    """
    p_num = r'(?<=[^A-Za-z_])[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|^[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
    p_str = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
    identifiers = set(re.findall(p_str, expr))
    if include_numbers:
        # only the number, not a + or -
        numbers = set(re.findall(p_num, expr))
    else:
        numbers = set()
    return (identifiers - KEYWORDS) | numbers


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
            # Just try to convert it to a sympy expression to get syntax errors
            # for incorrect expressions
            str_to_sympy(code)

        self.code = code.strip()
        self.identifiers = get_identifiers(code)

    @property
    def stochastic_variables(self):
        """
        Stochastic variables in this expression
        """
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
            return (self, None)

        stochastic_symbols = [sympy.Symbol(variable, real=True) for variable in stochastic_variables]

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
                                          'contains a stochastic symbol)' % (self.code, s_expr)))
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

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        return self.code == other.code

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.code)

    def __str__(self):
        return self.code

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.code)


class Equation(Hashable):
    """
    Class for internal use, encapsulates a single equation or parameter.

    .. note::
        This class should never be used directly, it is only useful as part of
        the `Equations` class.

    Parameters
    ----------
    type : {PARAMETER, DIFFERENTIAL_EQUATION, SUBEXPRESSION}
        The type of the equation.
    varname : str
        The variable that is defined by this equation.
    expr : `Expression`, optional
        The expression defining the variable (or ``None`` for parameters).
    flags: list of str, optional
        A list of flags that give additional information about this equation.
        What flags are possible depends on the type of the equation and the
        context.
    """

    def __init__(self, type, varname, expr=None, flags=None):
        self.type = type
        self.varname = varname
        self.expr = expr
        if flags is None:
            self.flags = []
        else:
            self.flags = flags

    @property
    def identifiers(self):
        """All identifiers in the RHS of this equation."""
        return self.expr.identifiers if not self.expr is None else set([])

    @property
    def stochastic_variables(self):
        """Stochastic variables in the RHS of this equation"""
        return set([variable for variable in self.identifiers
                    if variable == 'xi' or variable.startswith('xi_')])

    def __eq__(self, other):
        if not isinstance(other, Equation):
            raise NotImplementedError
        return self._state_tuple == other._state_tuple

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._state_tuple)

    def __str__(self):
        s = 'd' + self.varname + '/dt' if self.type == DIFFERENTIAL_EQUATION else self.varname
        if self.expr is not None:
            s += ' = ' + str(self.expr)
        if len(self.flags):
            s += ' : ' + ', '.join(self.flags)
        return s

    def __repr__(self):
        s = '<' + self.type + ' ' + self.varname
        if self.expr is not None:
            s += ': ' + self.expr.code
        if len(self.flags):
            s += ' (flags: ' + ', '.join(self.flags) + ')>'
        return s


class Equations(Hashable):
    """
    Container that stores equations from which models can be created.

    String equations can be of any of the following forms:

    1. ``dx/dt = f ( : flags)`` (differential equation)
    2. ``x = f ( : flags)`` (equation)
    3. ``x ( : flags)`` (parameter)

    String equations can span several lines and contain Python-style
    comments starting with ``#``

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
        self._eqs_str = eqns
        self._equations = process_eq_description(eqns)
        self.check_identifiers()

        # Check for special symbol xi (stochastic term)
        uses_xi = None
        e1 = 'The equation defining %s contains the symbol "xi" but is not a differential equation.'
        e2 = 'The equation defining %s contains the symbol "xi", but it is already used in the equation defining %s.'
        for eq in self._equations:
            if eq.expr is not None and 'xi' in eq.expr.identifiers:
                if not eq.type == DIFFERENTIAL_EQUATION:
                    raise ValueError(e1 % eq.varname)
                elif uses_xi is not None:
                    raise ValueError(e2 % (eq.varname, uses_xi))
                else:
                    uses_xi = eq.varname

        #: Cache for equations with the subexpressions substituted
        self._substituted_expressions = None

    def __iter__(self):
        return iter(self._equations)

    def __len__(self):
        return len(self._equations)

    def __getitem__(self, key):
        for eq in self._equations:
            if eq.varname == key:
                return eq

    def __add__(self, other_eqns):
        if isinstance(other_eqns, str):
            eqns = self._eqs_str + "\n" + other_eqns
            return Equations(eqns)
        else:
            raise NotImplemented

    def __hash__(self):
        return hash(frozenset(self._equations))

    def check_flags(self, allowed_flags, incompatible_flags=None):
        """
        Check the list of flags.

        Parameters
        ----------
        allowed_flags : dict
             A dictionary mapping equation types (PARAMETER,
             DIFFERENTIAL_EQUATION, SUBEXPRESSION) to a list of strings (the
             allowed flags for that equation type)
        incompatible_flags : list of tuple
            A list of flag combinations that are not allowed for the same
            equation.
        Notes
        -----
        Not specifying allowed flags for an equation type is the same as
        specifying an empty list for it.

        Raises
        ------
        ValueError
            If any flags are used that are not allowed.
        """
        if incompatible_flags is None:
            incompatible_flags = []

        for eq in self._equations:
            for flag in eq.flags:
                if eq.type not in allowed_flags or len(allowed_flags[eq.type]) == 0:
                    raise ValueError('Equations of type "%s" cannot have any flags.' % eq.type)
                if flag not in allowed_flags[eq.type]:
                    raise ValueError(('Equations of type "%s" cannot have a '
                                      'flag "%s", only the following flags '
                                      'are allowed: %s') % (eq.type,
                                                            flag, allowed_flags[eq.type]))
                # Check for incompatibilities
                for flag_combinations in incompatible_flags:
                    if flag in flag_combinations:
                        remaining_flags = set(flag_combinations) - set([flag])
                        for remaining_flag in remaining_flags:
                            if remaining_flag in eq.flags:
                                raise ValueError("Flag '{}' cannot be "
                                                 "combined with flag "
                                                 "'{}'".format(flag,
                                                               remaining_flag))

    def check_identifiers(self):
        """
        Check all identifiers for conformity with the rules.

        Raises
        ------
        ValueError
            If an identifier does not conform to the rules.

        See also
        --------
        Equations.check_identifier : The function that is called for each identifier.
        """
        for name in self.names:
            identifier = name

            # Check that an identifier is not using a reserved special variable name. The
            # special variables are: 't', 'dt', and 'xi', as well as everything starting
            # with `xi_`.
            keywords = ('t', 'dt', 'xi', 'i', 'N')
            if identifier in keywords or identifier.startswith('xi_'):
                raise SyntaxError(('"%s" has a special meaning in equations and cannot '
                                   'be used as a variable name.') % identifier)

            # Make sure that identifier names do not clash with function names.
            if identifier in DEFAULT_FUNCTIONS:
                raise SyntaxError('"%s" is the name of a function, cannot be used as a '
                                  'variable name.' % identifier)

            # Make sure that identifier names do not clash with function names.
            if identifier in DEFAULT_CONSTANTS:
                raise SyntaxError('"%s" is the name of a constant, cannot be used as a '
                                  'variable name.' % identifier)

    def get_substituted_expressions(self, variables=None):
        """
        Return a list of ``(varname, expr)`` tuples, containing all
        differential equations (and optionally subexpressions) with all the
        subexpression variables substituted with the respective expressions.

        Parameters
        ----------
        variables : dict, optional
            A mapping of variable names to `Variable`/`Function` objects.

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
            for eq in self._equations:
                # Skip parameters
                if eq.expr is None: continue

                new_sympy_expr = str_to_sympy(eq.expr.code, variables).xreplace(substitutions)
                new_str_expr = sympy_to_str(new_sympy_expr)
                expr = Expression(new_str_expr)

                if eq.type == SUBEXPRESSION:
                    substitutions.update({sympy.Symbol(eq.varname, real=True): str_to_sympy(expr.code, variables)})
                self._substituted_expressions.append((eq.varname, expr))

        return [(name, expr) for name, expr in self._substituted_expressions
                if self[name].type == DIFFERENTIAL_EQUATION]

    ############################################################################
    # Properties
    ############################################################################

    @property
    def diff_eq_expressions(self):
        """A list of (variable name, expression) tuples of all differential equations."""
        return [(eq.varname, eq.expr) for eq in self._equations
                if eq.type == DIFFERENTIAL_EQUATION]

    @property
    def eq_expressions(self):
        """A list of (variable name, expression) tuples of all equations."""
        return [(eq.varname, eq.expr) for eq in self._equations
                if eq.type in (SUBEXPRESSION, DIFFERENTIAL_EQUATION)]

    @property
    def names(self):
        """All variable names defined in the equations."""
        return set([eq.varname for eq in self._equations])

    @property
    def diff_eq_names(self):
        """All differential equation names."""
        return set([eq.varname for eq in self._equations if eq.type == DIFFERENTIAL_EQUATION])

    @property
    def subexpr_names(self):
        """All subexpression names."""
        return set([eq.varname for eq in self._equations if eq.type == SUBEXPRESSION])

    @property
    def eq_names(self):
        """All equation names (including subexpressions)."""
        return set([eq.varname for eq in self._equations if eq.type in (DIFFERENTIAL_EQUATION, SUBEXPRESSION)])

    @property
    def parameter_names(self):
        """All parameter names."""
        return set([eq.varname for eq in self._equations if eq.type == PARAMETER])

    @property
    def identifiers(self):
        """Set of all identifiers used in the equations,
        excluding the variables defined in the equations."""
        return set().union(*[eq.identifiers for eq in self._equations]) - self.names

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
    # Representation
    ############################################################################

    def __str__(self):
        strings = [str(eq) for eq in self._equations]
        return '\n'.join(strings)

    def __repr__(self):
        return '<Equations object consisting of %d equations>' % len(self._equations)


def process_eq_description(eq_desc):
    """Parse a string defining equations.

    Parameters
    ----------
    eq_desc : str
        The (possibly multi-line) string defining the equations. See the
        documentation of the `Equations` class for details.

    Returns
    -------
    equations : list
        A list contains instances of ``Equation``.
    """
    eq_desc = eq_desc.replace(';', '\n').split('\n')

    equations = []
    # Iterate over all lines
    for line in eq_desc:
        # Skip empty lines
        desc = line.strip()
        if desc == '':
            continue

        # Remove comments
        com = desc.split('#')
        if len(com) > 1:
            desc = com[0]
            if desc.strip() == '':
                continue

        # Process the line
        try:
            equation, flags = desc.rsplit(':', 1)
        except ValueError:  # There is no :, only equation is concerned
            equation = desc
            flags = []
        else:  # there is a :
            equation = equation.strip()
            flags = flags.strip()
            flags = [fl.strip() for fl in flags.split(',') if fl.strip()]

        # Split the equation around operators = += -= *= /=, but not ==
        try:
            parsed = (P_DIFF_EQ | P_STATIC_EQ).parseString(equation)
        except Exception:
            parsed = P_PARAMETER_EQ.parseString(equation)

        eq = parsed[0]
        eq_type = eq.getName()
        identifier = eq['identifier']
        expression = eq.get('expression')
        if expression is not None:
            # Replace multiple whitespaces (arising from
            # joining multiline strings) with single space
            p = re.compile(r'\s{2,}')
            expression = Expression(p.sub(' ', expression))

        eqs = Equation(eq_type, identifier, expr=expression, flags=flags)

        equations.append(eqs)

    return equations


def extract_method_options(method_options, default_options):
    """
    Helper function to check ``method_options`` against options understood by
    this state updater, and setting default values for all unspecified options.

    Parameters
    ----------
    method_options : dict or None
        The options that the user specified for the state update.
    default_options : dict
        The default option values for this state updater (each admissible option
        needs to be present in this dictionary). To specify that a state updater
        does not take any options, provide an empty dictionary as the argument.

    Returns
    -------
    options : dict
        The final dictionary with all the options either at their default or at
        the user-specified value.

    Raises
    ------
    KeyError
        If the user specifies an option that is not understood by this state
        updater.

    Examples
    --------
    # doctest: +IGNORE_EXCEPTION_DETAIL
    >>> options = extract_method_options({'a': True}, default_options={'b': False, 'c': False})
    Traceback (most recent call last):
    ...
    KeyError: 'method_options specifies "a", but this is not an option for
    this state updater. Avalaible options are: "b", "c".'

    # doctest: +IGNORE_EXCEPTION_DETAIL
    >>> options = extract_method_options({'a': True}, default_options={})
    Traceback (most recent call last):
    ...
    KeyError: 'method_options specifies "a", but this is not an option for this
    state updater. This state updater does not accept any options.'
    >>> options = extract_method_options({'a': True}, default_options={'a': False, 'b': False})
    >>> sorted(options.items())
    [('a', True), ('b', False)]
    """
    if method_options is None:
        method_options = {}
    for key in method_options:
        if key not in default_options:
            if len(default_options):
                keys = sorted(default_options.keys())
                options = 'Available options are: ' + ', '.join('"%s"' % key for key in keys) + '.'
            else:
                options = 'This state updater does not accept any options.'
            raise KeyError('method_options specifies "{key}", but this is not an option for this '
                           'state updater. {options}'.format(key=key, options=options))
    filled_options = dict(default_options)
    filled_options.update(method_options)
    return filled_options


class StateUpdate(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, equations, variables=None, method_options=None):
        """
        Generate abstract code from equations. The method also gets the
        the variables because some state updaters have to check whether
        variable names reflect other state variables (which can change from
        timestep to timestep) or are external values (which stay constant during
        a run)  For convenience, this arguments are optional -- this allows to
        directly see what code a state updater generates for a set of equations
        by simply writing ``euler(eqs)``, for example.

        Parameters
        ----------
        equations : `Equations`
            The model equations.
        variables : dict, optional
            The `Variable` objects for the model variables.
        method_options : dict, optional
            Additional options specific to the state updater.
        Returns
        -------
        code : str
            The abstract code performing a state update step.
        """
        pass

# ===============================================================================
# Class for simple definition of explicit state updaters
# ===============================================================================

#: reserved standard symbols
SYMBOLS = {
    '__x': sympy.Symbol('__x', real=True),
    '__t': sympy.Symbol('__t', real=True, positive=True),
    'dt': sympy.Symbol('dt', real=True, positive=True),
    't': sympy.Symbol('t', real=True, positive=True),
    '__f': sympy.Function('__f'),
    '__g': sympy.Function('__g'),
    '__dW': sympy.Symbol('__dW', real=True)
}


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


def split_expression(expr):
    """
    Split an expression into a part containing the function ``f`` and another
    one containing the function ``g``. Returns a tuple of the two expressions
    (as sympy expressions).

    Parameters
    ----------
    expr : str
        An expression containing references to functions ``f`` and ``g``.

    Returns
    -------
    (non_stochastic, stochastic) : tuple of sympy expressions
        A pair of expressions representing the non-stochastic (containing
        function-independent terms and terms involving ``f``) and the
        stochastic part of the expression (terms involving ``g`` and/or ``dW``).

    Examples
    --------
    >>> split_expression('dt * __f(__x, __t)')
    (dt*__f(__x, __t), None)
    >>> split_expression('dt * __f(__x, __t) + __dW * __g(__x, __t)')
    (dt*__f(__x, __t), __dW*__g(__x, __t))
    >>> split_expression('1/(2*dt**.5)*(__g_support - __g(__x, __t))*(__dW**2)')
    (0, __dW**2*__g_support*dt**(-0.5)/2 - __dW**2*dt**(-0.5)*__g(__x, __t)/2)
    """

    f = SYMBOLS['__f']
    g = SYMBOLS['__g']
    dW = SYMBOLS['__dW']
    # Arguments of the f and g functions
    x_f = sympy.Wild('x_f', exclude=[f, g], real=True)
    t_f = sympy.Wild('t_f', exclude=[f, g], real=True)
    x_g = sympy.Wild('x_g', exclude=[f, g], real=True)
    t_g = sympy.Wild('t_g', exclude=[f, g], real=True)

    # Reorder the expression so that f(x,t) and g(x,t) are factored out
    sympy_expr = sympy.sympify(expr, locals=SYMBOLS).expand()
    sympy_expr = sympy.collect(sympy_expr, f(x_f, t_f))
    sympy_expr = sympy.collect(sympy_expr, g(x_g, t_g))

    # Constant part, contains neither f, g nor dW
    independent = sympy.Wild('independent', exclude=[f, g, dW], real=True)
    # The exponent of the random number
    dW_exponent = sympy.Wild('dW_exponent', exclude=[f, g, dW, 0], real=True)
    # The factor for the random number, not containing the g function
    independent_dW = sympy.Wild('independent_dW', exclude=[f, g, dW], real=True)
    # The factor for the f function
    f_factor = sympy.Wild('f_factor', exclude=[f, g], real=True)
    # The factor for the g function
    g_factor = sympy.Wild('g_factor', exclude=[f, g], real=True)

    match_expr = (independent + f_factor * f(x_f, t_f) +
                  independent_dW * dW ** dW_exponent + g_factor * g(x_g, t_g))
    matches = sympy_expr.match(match_expr)

    if matches is None:
        raise ValueError('Expression "%s" in the state updater description '
                         'could not be parsed.' % sympy_expr)

    # Non-stochastic part
    if x_f in matches:
        # Includes the f function
        non_stochastic = matches[independent] + (matches[f_factor] * f(matches[x_f], matches[t_f]))
    else:
        # Does not include f, might be 0
        non_stochastic = matches[independent]

    # Stochastic part
    if independent_dW in matches and matches[independent_dW] != 0:
        # includes a random variable term with a non-zero factor
        stochastic = (matches[g_factor] * g(matches[x_g], matches[t_g]) +
                      matches[independent_dW] * dW ** matches[dW_exponent])
    elif x_g in matches:
        # Does not include a random variable but the g function
        stochastic = matches[g_factor] * g(matches[x_g], matches[t_g])
    else:
        # Contains neither random variable nor g function --> empty
        stochastic = None

    return non_stochastic, stochastic


class ExplicitStateUpdater(StateUpdate):
    """
    An object that can be used for defining state updaters via a simple
    description (see below). Resulting instances can be passed to the
    ``method`` argument of the `NeuronGroup` constructor. As other state
    updater functions the `ExplicitStateUpdater` objects are callable,
    returning abstract code when called with an `Equations` object.

    A description of an explicit state updater consists of a (multi-line)
    string, containing assignments to variables and a final "x_new = ...",
    stating the integration result for a single timestep. The assignments
    can be used to define an arbitrary number of intermediate results and
    can refer to ``f(x, t)`` (the function being integrated, as a function of
    ``x``, the previous value of the state variable and ``t``, the time) and
    ``dt``, the size of the timestep.

    For example, to define a Runge-Kutta 4 integrator (already provided as
    `rk4`), use::

            k1 = dt*f(x,t)
            k2 = dt*f(x+k1/2,t+dt/2)
            k3 = dt*f(x+k2/2,t+dt/2)
            k4 = dt*f(x+k3,t+dt)
            x_new = x+(k1+2*k2+2*k3+k4)/6

    Note that for stochastic equations, the function `f` only corresponds to
    the non-stochastic part of the equation. The additional function `g`
    corresponds to the stochastic part that has to be multiplied with the
    stochastic variable xi (a standard normal random variable -- if the
    algorithm needs a random variable with a different variance/mean you have
    to multiply/add it accordingly). Equations with more than one
    stochastic variable do not have to be treated differently, the part
    referring to ``g`` is repeated for all stochastic variables automatically.

    Stochastic integrators can also make reference to ``dW`` (a normal
    distributed random number with variance ``dt``) and ``g(x, t)``, the
    stochastic part of an equation. A stochastic state updater could therefore
    use a description like::

        x_new = x + dt*f(x,t) + g(x, t) * dW

    For simplicity, the same syntax is used for state updaters that only support
    additive noise, even though ``g(x, t)`` does not depend on ``x`` or ``t``
    in that case.

    There a some restrictions on the complexity of the expressions (but most
    can be worked around by using intermediate results as in the above Runge-
    Kutta example): Every statement can only contain the functions ``f`` and
    ``g`` once; The expressions have to be linear in the functions, e.g. you
    can use ``dt*f(x, t)`` but not ``f(x, t)**2``.

    Parameters
    ----------
    description : str
        A state updater description (see above).
    stochastic : {None, 'additive', 'multiplicative'}
        What kind of stochastic equations this state updater supports: ``None``
        means no support of stochastic equations, ``'additive'`` means only
        equations with additive noise and ``'multiplicative'`` means
        supporting arbitrary stochastic equations.

    Raises
    ------
    ValueError
        If the parsing of the description failed.

    Notes
    -----
    Since clocks are updated *after* the state update, the time ``t`` used
    in the state update step is still at its previous value. Enumerating the
    states and discrete times, ``x_new = x + dt*f(x, t)`` is therefore
    understood as :math:`x_{i+1} = x_i + dt f(x_i, t_i)`, yielding the correct
    forward Euler integration. If the integrator has to refer to the time at
    the end of the timestep, simply use ``t + dt`` instead of ``t``.

    See also
    --------
    euler, rk2, rk4, milstein
    """

    # ===========================================================================
    # Parsing definitions
    # ===========================================================================

    #: A single expression
    P_EXPRESSION = restOfLine.setResultsName('expression')

    #: Legal names for temporary variables
    P_TEMP_VAR = ~Literal('x_new') + Word(alphas + '_', alphas + nums + '_').setResultsName('identifier')
    #: An assignment statement
    P_STATEMENT = Group(P_TEMP_VAR + Suppress('=') + P_EXPRESSION).setResultsName('statement')

    #: The last line of a state updater description
    P_OUTPUT = Group(Suppress(Literal('x_new')) + Suppress('=') + P_EXPRESSION).setResultsName('output')

    #: A complete state updater description
    P_DESCRIPTION = ZeroOrMore(P_STATEMENT) + P_OUTPUT

    def __init__(self, description, stochastic=None, custom_check=None):
        self.description = description
        self.stochastic = stochastic
        self.custom_check = custom_check
        self.statements = []
        self.symbols = SYMBOLS.copy()

        try:
            parsed = ExplicitStateUpdater.P_DESCRIPTION.parseString(description, parseAll=True)
        except ParseException as p_exc:
            ex = SyntaxError('Parsing failed: ' + str(p_exc.msg))
            ex.text = str(p_exc.line)
            ex.offset = p_exc.column
            ex.lineno = p_exc.lineno
            raise ex

        for element in parsed:
            expression = str_to_sympy(element.expression)
            # Replace all symbols used in state updater expressions by unique
            # names that cannot clash with user-defined variables or functions
            expression = expression.subs(sympy.Function('f'), self.symbols['__f'])
            expression = expression.subs(sympy.Function('g'), self.symbols['__g'])
            symbols = list(expression.atoms(sympy.Symbol))
            unique_symbols = []
            for symbol in symbols:
                if symbol.name == 'dt':
                    unique_symbols.append(symbol)
                else:
                    unique_symbols.append(sympy.Symbol('__' + symbol.name, real=True))
            for symbol, unique_symbol in zip(symbols, unique_symbols):
                expression = expression.subs(symbol, unique_symbol)

            self.symbols.update(dict(((symbol.name, symbol) for symbol in unique_symbols)))
            if element.getName() == 'statement':
                self.statements.append(('__' + element.identifier, expression))
            elif element.getName() == 'output':
                self.output = expression
            else:
                raise AssertionError('Unknown element name: %s' % element.getName())
        print()

    def __repr__(self):
        description = '\n'.join(['%s = %s' % (var, expr) for var, expr in self.statements])
        if len(description):
            description += '\n'
        description += 'x_new = ' + str(self.output)
        r = "{classname}('''{description}''', stochastic={stochastic})"
        return r.format(classname=self.__class__.__name__, description=description, stochastic=repr(self.stochastic))

    def __str__(self):
        s = '%s\n' % self.__class__.__name__
        if len(self.statements) > 0:
            s += 'Intermediate statements:\n'
            s += '\n'.join([(var + ' = ' + sympy_to_str(expr)) for var, expr in self.statements])
            s += '\n'
        s += 'Output:\n'
        s += sympy_to_str(self.output)
        return s

    def replace_func(self, x, t, expr, temp_vars, eq_symbols, stochastic_variable=None):
        """
        Used to replace a single occurrence of ``f(x, t)`` or ``g(x, t)``:
        `expr` is the non-stochastic (in the case of ``f``) or stochastic
        part (``g``) of the expression defining the right-hand-side of the
        differential equation describing `var`. It replaces the variable
        `var` with the value given as `x` and `t` by the value given for
        `t`. Intermediate variables will be replaced with the appropriate
        replacements as well.

        For example, in the `rk2` integrator, the second step involves the
        calculation of ``f(k/2 + x, dt/2 + t)``.  If `var` is ``v`` and
        `expr` is ``-v / tau``, this will result in ``-(_k_v/2 + v)/tau``.

        Note that this deals with only one state variable `var`, given as
        an argument to the surrounding `_generate_RHS` function.
        """

        try:
            s_expr = str_to_sympy(str(expr))
        except SympifyError as ex:
            raise ValueError('Error parsing the expression "%s": %s' % (expr, str(ex)))

        for var in eq_symbols:
            # Generate specific temporary variables for the state variable,
            # e.g. '_k_v' for the state variable 'v' and the temporary
            # variable 'k'.
            if stochastic_variable is None:
                temp_var_replacements = dict(
                    ((self.symbols[temp_var], sympy.Symbol(temp_var + '_' + var, real=True))
                     for temp_var in temp_vars)
                )
            else:
                temp_var_replacements = dict(
                    ((self.symbols[temp_var],
                      sympy.Symbol(temp_var + '_' + var + '_' + stochastic_variable, real=True))
                     for temp_var in temp_vars)
                )
            # In the expression given as 'x', replace 'x' by the variable
            # 'var' and all the temporary variables by their
            # variable-specific counterparts.
            x_replacement = x.subs(self.symbols['__x'], eq_symbols[var])
            x_replacement = x_replacement.subs(temp_var_replacements)

            # Replace the variable `var` in the expression by the new `x`
            # expression
            s_expr = s_expr.subs(eq_symbols[var], x_replacement)

        # If the expression given for t in the state updater description
        # is not just "t" (or rather "__t"), then replace t in the
        # equations by it, and replace "__t" by "t" afterwards.
        if t != self.symbols['__t']:
            s_expr = s_expr.subs(SYMBOLS['t'], t)
            s_expr = s_expr.replace(self.symbols['__t'], SYMBOLS['t'])

        return s_expr

    def _non_stochastic_part(self, eq_symbols, non_stochastic, non_stochastic_expr, stochastic_variable,
                             temp_vars, var):
        non_stochastic_results = []
        if stochastic_variable is None or len(stochastic_variable) == 0:
            # Replace the f(x, t) part
            replace_f = lambda x, t: self.replace_func(x, t, non_stochastic, temp_vars, eq_symbols)
            non_stochastic_result = non_stochastic_expr.replace(self.symbols['__f'], replace_f)
            # Replace x by the respective variable
            non_stochastic_result = non_stochastic_result.subs(self.symbols['__x'], eq_symbols[var])
            # Replace intermediate variables
            temp_var_replacements = dict(
                (self.symbols[temp_var],
                 sympy.Symbol(temp_var + '_' + var, real=True))
                for temp_var in temp_vars
            )
            non_stochastic_result = non_stochastic_result.subs(temp_var_replacements)
            non_stochastic_results.append(non_stochastic_result)
        elif isinstance(stochastic_variable, str):
            # Replace the f(x, t) part
            replace_f = lambda x, t: self.replace_func(x, t, non_stochastic, temp_vars, eq_symbols, stochastic_variable)
            non_stochastic_result = non_stochastic_expr.replace(self.symbols['__f'], replace_f)
            # Replace x by the respective variable
            non_stochastic_result = non_stochastic_result.subs(self.symbols['__x'], eq_symbols[var])
            # Replace intermediate variables
            temp_var_replacements = dict(
                (self.symbols[temp_var],
                 sympy.Symbol(temp_var + '_' + var + '_' + stochastic_variable, real=True))
                for temp_var in temp_vars
            )

            non_stochastic_result = non_stochastic_result.subs(temp_var_replacements)
            non_stochastic_results.append(non_stochastic_result)
        else:
            # Replace the f(x, t) part
            replace_f = lambda x, t: self.replace_func(x, t, non_stochastic, temp_vars, eq_symbols)
            non_stochastic_result = non_stochastic_expr.replace(self.symbols['__f'], replace_f)
            # Replace x by the respective variable
            non_stochastic_result = non_stochastic_result.subs(self.symbols['__x'], eq_symbols[var])
            # Replace intermediate variables
            temp_var_replacements = dict(
                (self.symbols[temp_var],
                 reduce(operator.add, [sympy.Symbol(temp_var + '_' + var + '_' + xi, real=True)
                                       for xi in stochastic_variable]))
                for temp_var in temp_vars
            )

            non_stochastic_result = non_stochastic_result.subs(temp_var_replacements)
            non_stochastic_results.append(non_stochastic_result)

        return non_stochastic_results

    def _stochastic_part(self, eq_symbols, stochastic, stochastic_expr,
                         stochastic_variable, temp_vars, var):
        stochastic_results = []
        if isinstance(stochastic_variable, str):
            # Replace the g(x, t) part
            replace_f = lambda x, t: self.replace_func(x, t, stochastic.get(stochastic_variable, 0),
                                                       temp_vars, eq_symbols, stochastic_variable)
            stochastic_result = stochastic_expr.replace(self.symbols['__g'], replace_f)
            # Replace x by the respective variable
            stochastic_result = stochastic_result.subs(self.symbols['__x'], eq_symbols[var])
            # Replace dW by the respective variable
            stochastic_result = stochastic_result.subs(self.symbols['__dW'], stochastic_variable)
            # Replace intermediate variables
            temp_var_replacements = dict(
                (self.symbols[temp_var],
                 sympy.Symbol(temp_var + '_' + var + '_' + stochastic_variable, real=True))
                for temp_var in temp_vars
            )

            stochastic_result = stochastic_result.subs(temp_var_replacements)
            stochastic_results.append(stochastic_result)
        else:
            for xi in stochastic_variable:
                # Replace the g(x, t) part
                replace_f = lambda x, t: self.replace_func(x, t, stochastic.get(xi, 0),
                                                           temp_vars, eq_symbols, xi)
                stochastic_result = stochastic_expr.replace(self.symbols['__g'], replace_f)
                # Replace x by the respective variable
                stochastic_result = stochastic_result.subs(self.symbols['__x'], eq_symbols[var])

                # Replace dW by the respective variable
                stochastic_result = stochastic_result.subs(self.symbols['__dW'], xi)

                # Replace intermediate variables
                temp_var_replacements = dict(
                    (self.symbols[temp_var],
                     sympy.Symbol(temp_var + '_' + var + '_' + xi, real=True))
                    for temp_var in temp_vars
                )

                stochastic_result = stochastic_result.subs(temp_var_replacements)
                stochastic_results.append(stochastic_result)
        return stochastic_results

    def _step_code(self, var, eq_symbols, temp_vars, expr, non_stochastic_expr,
                   stochastic_expr, stochastic_variable=()):
        """Generates the right hand side of an abstract code statement by appropriately
        replacing f, g and t.

        For example, given a differential equation
        ``dv/dt = -(v + I) / tau`` (i.e. `var` is ``v` and `expr` is ``(-v + I) / tau``)
        together with the `rk2` step ``return x + dt*f(x +  k/2, t + dt/2)``
        (i.e. `non_stochastic_expr` is ``x + dt*f(x +  k/2, t + dt/2)`` and
        `stochastic_expr` is ``None``), produces ``v + dt*(-v - _k_v/2 + I + _k_I/2)/tau``.
        """

        # Note: in the following we are silently ignoring the case that a
        # state updater does not care about either the non-stochastic or the
        # stochastic part of an equation. We do trust state updaters to
        # correctly specify their own abilities (i.e. they do not claim to
        # support stochastic equations but actually just ignore the stochastic
        # part). We can't really check the issue here, as we are only dealing
        # with one line of the state updater description. It is perfectly valid
        # to write the euler update as:
        #     non_stochastic = dt * f(x, t)
        #     stochastic = dt**.5 * g(x, t) * xi
        #     return x + non_stochastic + stochastic
        #
        # In the above case, we'll deal with lines which do not define either
        # the stochastic or the non-stochastic part.

        non_stochastic, stochastic = expr.split_stochastic()

        if non_stochastic_expr is not None:
            non_stochastic_results = self._non_stochastic_part(
                eq_symbols, non_stochastic, non_stochastic_expr, stochastic_variable, temp_vars, var)
        else:
            non_stochastic_results = []

        if not (stochastic is None or stochastic_expr is None):
            stochastic_results = self._stochastic_part(
                eq_symbols, stochastic, stochastic_expr, stochastic_variable, temp_vars, var)
        else:
            stochastic_results = []

        RHS = sympy.Number(0)
        # All the parts (one non-stochastic and potentially more than one
        # stochastic part) are combined with addition
        for non_stochastic_result in non_stochastic_results:
            RHS += non_stochastic_result
        for stochastic_result in stochastic_results:
            RHS += stochastic_result

        return sympy_to_str(RHS)

    def __call__(self, eqs, variables=None, size=1):
        """
        Apply a state updater description to model equations.

        Parameters
        ----------
        eqs : `Equations`
            The equations describing the model
        variables: dict-like, optional
            The `Variable` objects for the model. Ignored by the explicit
            state updater.

        Examples
        --------
        >>> eqs = Equations('dv/dt = -v / tau : volt')
        >>> print(euler(eqs))
        _v = -dt*v/tau + v
        v = _v
        >>> print(rk4(eqs))
        __k_1_v = -dt*v/tau
        __k_2_v = -dt*(0.5*__k_1_v + v)/tau
        __k_3_v = -dt*(0.5*__k_2_v + v)/tau
        __k_4_v = -dt*(__k_3_v + v)/tau
        _v = 0.166666666666667*__k_1_v + 0.333333333333333*__k_2_v + 0.333333333333333*__k_3_v + 0.166666666666667*__k_4_v + v
        v = _v
        """
        # Non-stochastic numerical integrators should work for all equations,
        # except for stochastic equations
        if eqs.is_stochastic and self.stochastic is None:
            raise Exception('Cannot integrate stochastic equations with this state updater.')
        if self.custom_check:
            self.custom_check(eqs, variables)

        # The stochastic variables
        stochastic_variables = eqs.stochastic_variables

        # The variables for the intermediate results in the state updater
        # description, e.g. the variable k in rk2
        inter_variables = [var for var, expr in self.statements]

        # A dictionary mapping all the variables in the equations to their
        # sympy representations
        eq_variables = dict([(var, sympy.Symbol(var, real=True)) for var in eqs.eq_names])

        diff_expressions = eqs.get_substituted_expressions(variables)
        # Generate the random numbers for the stochastic variables
        statements = []
        for stochastic_variable in stochastic_variables:
            statements.append(stochastic_variable + ' = ' + 'dt**.5 * bnp.random.randn({})'.format(size))
        # Process the intermediate statements in the stateupdater description
        for inter_var, inter_expr in self.statements:
            # Split the expression into a non-stochastic and a stochastic part
            non_stochastic_expr, stochastic_expr = split_expression(inter_expr)
            # Execute the statement by appropriately replacing the functions f
            # and g and the variable x for every equation in the model.
            # We use the model equations where the subexpressions have
            # already been substituted into the model equations.
            for var, expr in diff_expressions:
                if not stochastic_variables:  # no stochastic variables
                    RHS = self._step_code(var, eq_variables, inter_variables,
                                          expr, non_stochastic_expr, stochastic_expr)
                    statements.append(inter_var + '_' + var + ' = ' + RHS)
                else:
                    for xi in stochastic_variables:
                        RHS = self._step_code(var, eq_variables, inter_variables,
                                              expr, non_stochastic_expr, stochastic_expr, xi)
                        statements.append(inter_var + '_' + var + '_' + xi + ' = ' + RHS)

        if eqs.is_stochastic and self.stochastic != 'multiplicative' and eqs.stochastic_type == 'multiplicative':
            # The equations are marked as having multiplicative noise and the
            # current state updater does not support such equations. However,
            # it is possible that the equations do not use multiplicative noise
            # at all. They could depend on time via a function that is constant
            # over a single time step (most likely, a TimedArray). In that case
            # we can integrate the equations
            dt_value = variables['dt'].get_value()[0] if 'dt' in variables else None
            for _, expr in diff_expressions:
                _, stoch = expr.split_stochastic()
                if stoch is None: continue
                # There could be more than one stochastic variable (e.g. xi_1, xi_2)
                for _, stoch_expr in stoch.items():
                    sympy_expr = str_to_sympy(stoch_expr.code)
                    # The equation really has multiplicative noise, if it depends
                    # on time (and not only via a function that is constant
                    # over dt), or if it depends on another variable defined
                    # via differential equations.
                    if (not is_constant_over_dt(sympy_expr, variables, dt_value)
                            or len(stoch_expr.identifiers & eqs.diff_eq_names)):
                        raise Exception('Cannot integrate equations with multiplicative noise with this state updater.')

        # Process the "return" line of the stateupdater description
        non_stochastic_expr, stochastic_expr = split_expression(self.output)

        # Assign a value to all the model variables described
        # by differential equations
        for var, expr in diff_expressions:
            RHS = self._step_code(var, eq_variables, inter_variables,
                                  expr, non_stochastic_expr, stochastic_expr, stochastic_variables)
            statements.append('_' + var + ' = ' + RHS)

        # Assign everything to the final variables
        for var, expr in diff_expressions:
            statements.append(var + ' = ' + '_' + var)

        return '\n'.join(statements)


# ===============================================================================
# Excplicit state updaters
# ===============================================================================

# these objects can be used like functions because they are callable

#: Forward Euler state updater
euler = ExplicitStateUpdater('x_new = x + dt * f(x,t) + g(x,t) * dW',
                             stochastic='additive')

#: Second order Runge-Kutta method (midpoint method)
rk2 = ExplicitStateUpdater('''
    k = dt * f(x,t)
    x_new = x + dt*f(x +  k/2, t + dt/2)''')

#: Classical Runge-Kutta method (RK4)
rk4 = ExplicitStateUpdater('''
    k_1 = dt*f(x,t)
    k_2 = dt*f(x+k_1/2,t+dt/2)
    k_3 = dt*f(x+k_2/2,t+dt/2)
    k_4 = dt*f(x+k_3,t+dt)
    x_new = x+(k_1+2*k_2+2*k_3+k_4)/6
    ''')


def diagonal_noise(equations, variables):
    """
    Checks whether we deal with diagonal noise, i.e. one independent noise
    variable per variable.

    Raises
    ------
    UnsupportedEquationsException
        If the noise is not diagonal.
    """
    if not equations.is_stochastic:
        return

    stochastic_vars = []
    for _, expr in equations.get_substituted_expressions(variables):
        expr_stochastic_vars = expr.stochastic_variables
        if len(expr_stochastic_vars) > 1:
            # More than one stochastic variable --> no diagonal noise
            raise Exception('Cannot integrate stochastic equations with '
                            'non-diagonal noise with this state updater.')
        stochastic_vars.extend(expr_stochastic_vars)

    # If there's no stochastic variable is used in more than one equation, we
    # have diagonal noise
    if len(stochastic_vars) != len(set(stochastic_vars)):
        raise Exception('Cannot integrate stochastic equations with '
                        'non-diagonal noise with this state updater.')


#: Derivative-free Milstein method
milstein = ExplicitStateUpdater('''
    x_support = x + dt*f(x, t) + dt**.5 * g(x, t)
    g_support = g(x_support, t)
    k = 1/(2*dt**.5)*(g_support - g(x, t))*(dW**2)
    x_new = x + dt*f(x,t) + g(x, t) * dW + k
    ''', stochastic='multiplicative', custom_check=diagonal_noise)

#: Stochastic Heun method (for multiplicative Stratonovic SDEs with non-diagonal
#: diffusion matrix)
heun = ExplicitStateUpdater('''
    x_support = x + g(x,t) * dW
    g_support = g(x_support,t+dt)
    x_new = x + dt*f(x,t) + .5*dW*(g(x,t)+g_support)
    ''', stochastic='multiplicative')


def exponential_euler_updater(equations, variables=None, method_options=None):
    """
    A state updater for conditionally linear equations, i.e. equations where
    each variable only depends linearly on itself (but possibly non-linearly
    on other variables). Typical Hodgkin-Huxley equations fall into this
    category, it is therefore the default integration method used in the
    GENESIS simulator, for example.
    """
    if equations.is_stochastic:
        raise ValueError('Cannot solve stochastic equations with this state updater.')

    # Convert equations into a linear system using sympy #
    # -------------------------------------------------- #
    # For examples:
    # >>> eqs = Equations('''
    #       dv/dt = (-v + w**2) / tau
    #       dw/dt = -w / tau
    #       du/dt = -v / tau
    #     )
    # >>> system = get_conditionally_linear_system(eqs)
    # >>> print(system['v'])
    #   (-1/tau, w**2.0/tau)
    # >>> print(system['w'])
    #   (-1/tau, 0)
    # >>> print(system['u'])
    #   (0, -v / tau)
    coefficients = {}
    diff_eqs = eqs.get_substituted_expressions(variables)
    for name, expr in diff_eqs:
        var = sympy.Symbol(name, real=True)
        s_expr = str_to_sympy(expr.code, variables).expand()
        if s_expr.has(var):
            s_expr = sympy.collect(s_expr, var, evaluate=False)
            if len(s_expr) > 2 or var not in s_expr:
                raise ValueError('The expression "%s", defining the variable %s, could not '
                                 'be separated into linear components' % (expr, name))
            coefficients[name] = (s_expr[var], s_expr.get(1, 0))
        else:
            coefficients[name] = (0, s_expr)
    system = coefficients

    # format the codes
    code = []
    for var, (A, B) in system.items():
        s_var = sympy.Symbol(var)
        s_dt = sympy.Symbol('dt')
        if A == 0:
            update_expression = s_var + s_dt * B
        elif B != 0:
            BA = B / A
            BA_name = '_BA_' + var
            BA_symbol = sympy.Symbol(BA_name)
            code += [BA_name + ' = ' + sympy_to_str(BA)]
            update_expression = (s_var + BA_symbol) * sympy.exp(A * s_dt) - BA_symbol
        else:
            update_expression = s_var * sympy.exp(A * s_dt)
        # The actual update step
        update = '_{var} = {expr}'
        code += [update.format(var=var, expr=sympy_to_str(update_expression))]

    # Replace all the variables with their updated value
    for var in system:
        code += ['{var} = _{var}'.format(var=var)]

    return '\n'.join(code)


if __name__ == '__main__1':
    # eqs = Equations('dv/dt = -v / tau : volt')
    eqs = Equations('dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau')
    print(euler(eqs))
    print('-' * 20)
    print(milstein(eqs))


if __name__ == '__main__':
    eqs = Equations("""
            a 
            dn/dt = an * (1.0 - n) - bn * n : init = 0.3, midpoint
            dm/dt = am * (1.0 - m) - bm * m : init = 0.0, midpoint
            dh/dt = ah * (1.0 - h) - bh * h : init = 0.6, midpoint
        """)
    # print(euler(eqs))
    # print('-' * 20)
    # print(milstein(eqs))
    # print('-' * 20)
    print(rk4(eqs))


if __name__ == '__main__1':
    des = """
        a 
        dn/dt = an * (1.0 - n) - bn * n : init = 0.3, midpoint
        dm/dt = am * (1.0 - m) - bm * m : init = 0.0, midpoint
        dh/dt = ah * (1.0 - h) - bh * h : init = 0.6, midpoint
    """
    variable_list = process_eq_description(des)
    print(variable_list)


if __name__ == '__main__1':
    eqs = Equations('''
        dv/dt = (-v + w**2) / tau : 1
        dw/dt = -w / tau : 1
        du/dt = -v / tau : 1
    ''')
    print(exponential_euler_updater(eqs))

