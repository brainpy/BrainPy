# -*- coding: utf-8 -*-

import sympy.functions.elementary.complexes
import sympy.functions.elementary.exponential
import sympy.functions.elementary.hyperbolic
import sympy.functions.elementary.integers
import sympy.functions.elementary.miscellaneous
import sympy.functions.elementary.trigonometric
from sympy.codegen import cfunctions

DEFAULT_FUNCTIONS = {
    'real': sympy.functions.elementary.complexes.re,
    'imag': sympy.functions.elementary.complexes.im,
    'conjugate': sympy.functions.elementary.complexes.conjugate,
    'sign': sympy.sign,
    'abs': sympy.functions.elementary.complexes.Abs,

    'cos': sympy.functions.elementary.trigonometric.cos,
    'sin': sympy.functions.elementary.trigonometric.sin,
    'tan': sympy.functions.elementary.trigonometric.tan,
    'sinc': sympy.functions.elementary.trigonometric.sinc,
    'arcsin': sympy.functions.elementary.trigonometric.asin,
    'arccos': sympy.functions.elementary.trigonometric.acos,
    'arctan': sympy.functions.elementary.trigonometric.atan,
    'arctan2': sympy.functions.elementary.trigonometric.atan2,

    'cosh': sympy.functions.elementary.hyperbolic.cosh,
    'sinh': sympy.functions.elementary.hyperbolic.sinh,
    'tanh': sympy.functions.elementary.hyperbolic.tanh,
    'arcsinh': sympy.functions.elementary.hyperbolic.asinh,
    'arccosh': sympy.functions.elementary.hyperbolic.acosh,
    'arctanh': sympy.functions.elementary.hyperbolic.atanh,

    'ceil': sympy.functions.elementary.integers.ceiling,
    'floor': sympy.functions.elementary.integers.floor,

    'log': sympy.functions.elementary.exponential.log,
    'log2': cfunctions.log2,
    'log1p': cfunctions.log1p,
    'log10': cfunctions.log10,
    'exp': sympy.functions.elementary.exponential.exp,
    'expm1': cfunctions.expm1,
    'exp2': cfunctions.exp2,
    'hypot': cfunctions.hypot,

    'sqrt': sympy.functions.elementary.miscellaneous.sqrt,
    'min': sympy.functions.elementary.miscellaneous.Min,
    'max': sympy.functions.elementary.miscellaneous.Max,
    'cbrt': sympy.functions.elementary.miscellaneous.cbrt,
}

DEFAULT_CONSTANTS = {'pi': sympy.pi,
                     'e': sympy.E,
                     'inf': sympy.S.Infinity,
                     '-inf': sympy.S.NegativeInfinity}
