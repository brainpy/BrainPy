# -*- coding: utf-8 -*-


CONSTANT_NOISE = 'CONSTANT'
FUNCTIONAL_NOISE = 'FUNCTIONAL'

_ODE_TYPE = 'ODE'
_SDE_TYPE = 'SDE'

_DIFF_EQUATION = 'diff_equation'
_SUB_EXPRESSION = 'sub_expression'

RETURN_TYPES = [
    # return type      # multi_return     # DF type
    'x',               # False            # ODE
    'x,x',             # False            # SDE
    '(x,),',           # False            # ODE
    '(x,),...',        # True             # ODE
    '(x,x),',          # False            # SDE
    '(x,x),...',       # True             # SDE
]
