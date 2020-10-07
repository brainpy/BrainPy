import itertools
import logging

import sympy

from npbrain.integration._equations import is_constant_over_dt
from npbrain.integration.sympy_tools import sympy_to_str, str_to_sympy
from npbrain.integration.base import StateUpdateMethod
from npbrain.integration.base import EquationError
from npbrain.integration.base import extract_method_options

__all__ = ['exact', ]

logger = logging.Logger(__name__)


def get_linear_system(eqs, variables):
    """
    Convert equations into a linear system using sympy.
    
    Parameters
    ----------
    eqs : `Equations`
        The model equations.
    variables : Optional
    
    Returns
    -------
    (diff_eq_names, coefficients, constants) : (list of str, `sympy.Matrix`, `sympy.Matrix`)
        A tuple containing the variable names (`diff_eq_names`) corresponding
        to the rows of the matrix `coefficients` and the vector `constants`,
        representing the system of equations in the form M * X + B
    
    Raises
    ------
    ValueError
        If the equations cannot be converted into an M * X + B form.
    """
    diff_eqs = eqs.substitute(variables)
    diff_eq_names = [name for name, _ in diff_eqs]

    symbols = [sympy.Symbol(name, real=True) for name in diff_eq_names]

    coefficients = sympy.zeros(len(diff_eq_names))
    constants = sympy.zeros(len(diff_eq_names), 1)

    for row_idx, (name, expr) in enumerate(diff_eqs):
        s_expr = str_to_sympy(expr.code, variables).expand()

        current_s_expr = s_expr
        for col_idx, symbol in enumerate(symbols):
            current_s_expr = current_s_expr.collect(symbol)
            constant_wildcard = sympy.Wild('c', exclude=[symbol])
            factor_wildcard = sympy.Wild('c_' + name, exclude=symbols)
            one_pattern = factor_wildcard * symbol + constant_wildcard
            matches = current_s_expr.match(one_pattern)
            if matches is None:
                raise EquationError(('The expression "%s", '
                                'defining the variable '
                                '%s, could not be '
                                'separated into linear '
                                'components.') %
                                    (expr, name))

            coefficients[row_idx, col_idx] = matches[factor_wildcard]
            current_s_expr = matches[constant_wildcard]

        # The remaining constant should be a true constant
        constants[row_idx] = current_s_expr

    return diff_eq_names, coefficients, constants


class LinearStateUpdater(StateUpdateMethod):
    """
    A state updater for linear equations. Derives a state updater step from the
    analytical solution given by sympy. Uses the matrix exponential (which is
    only implemented for diagonalizable matrices in sympy).
    """

    def __call__(self, equations, variables=None, method_options=None):
        method_options = extract_method_options(method_options, {'simplify': True})

        if equations.is_stochastic:
            raise EquationError('Cannot solve stochastic '
                           'equations with this state '
                           'updater.')
        if variables is None:
            variables = {}

        # Get a representation of the ODE system in the form of
        # dX/dt = M*X + B
        varnames, matrix, constants = get_linear_system(equations, variables)

        # No differential equations, nothing to do (this occurs sometimes in the
        # test suite where the whole model is nothing more than something like
        # 'v : 1')
        if matrix.shape == (0, 0):
            return ''

        # Make sure that the matrix M is constant, i.e. it only contains
        # external variables or constant variables
        t = sympy.Symbol('t', real=True, positive=True)

        # Check for time dependence
        dt_value = variables['dt'].get_value()[0] if 'dt' in variables else None

        # This will raise an error if we meet the symbol "t" anywhere
        # except as an argument of a locally constant function
        for entry in itertools.chain(matrix, constants):
            if not is_constant_over_dt(entry, variables, dt_value):
                raise EquationError(
                    ('Expression "{}" is not guaranteed to be constant over a '
                     'time step').format(sympy_to_str(entry)))

        symbols = [sympy.Symbol(variable, real=True) for variable in varnames]
        solution = sympy.solve_linear_system(matrix.row_join(constants), *symbols)
        if solution is None or set(symbols) != set(solution.keys()):
            raise EquationError('Cannot solve the given '
                           'equations with this '
                           'stateupdater.')
        b = sympy.ImmutableMatrix([solution[symbol] for symbol in symbols])

        # Solve the system
        dt = sympy.Symbol('dt', real=True, positive=True)
        try:
            A = (matrix * dt).exp()
        except NotImplementedError:
            raise EquationError('Cannot solve the given '
                           'equations with this '
                           'stateupdater.')
        if method_options['simplify']:
            A = A.applyfunc(lambda x: sympy.factor_terms(sympy.cancel(sympy.signsimp(x))))
        C = sympy.ImmutableMatrix(A * b) - b
        _S = sympy.MatrixSymbol('_S', len(varnames), 1)
        updates = A * _S + C
        updates = updates.as_explicit()

        # The solution contains _S[0, 0], _S[1, 0] etc. for the state variables,
        # replace them with the state variable names 
        abstract_code = []
        for idx, (variable, update) in enumerate(zip(varnames, updates)):
            rhs = update
            if rhs.has(sympy.I, sympy.re, sympy.im):
                raise EquationError('The solution to the linear system '
                               'contains complex values '
                               'which is currently not implemented.')
            for row_idx, varname in enumerate(varnames):
                rhs = rhs.subs(_S[row_idx, 0], varname)

            # Do not overwrite the real state variables yet, the update step
            # of other state variables might still need the original values
            abstract_code.append('_' + variable + ' = ' + sympy_to_str(rhs))

        # Update the state variables
        for variable in varnames:
            abstract_code.append('{variable} = _{variable}'.format(variable=variable))
        return '\n'.join(abstract_code)

    def __repr__(self):
        return '%s()' % self.__class__.__name__


exact = LinearStateUpdater()
