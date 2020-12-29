# -*- coding: utf-8 -*-


CONSTANT_NOISE = 'CONSTANT'
FUNCTIONAL_NOISE = 'FUNCTIONAL'

ODE_TYPE = 'ODE'
SDE_TYPE = 'SDE'

DIFF_EQUATION = 'diff_equation'
SUB_EXPRESSION = 'sub_expression'
RETURN_TYPES = [
    # return type      # multi_return     # DF type
    'x',               # False            # ODE
    'x,0',             # False            # ODE [x]
    '(x,),',           # False            # ODE
    '(x,),...',        # True             # ODE
    '(x,0),',          # False            # ODE [x]
    '(x,0),...',       # True             # ODE [x]
    'x,x',             # False            # SDE
    '(x,x),',          # False            # SDE
    '(x,x),...',       # True             # SDE
]

