# -*- coding: utf-8 -*-


__all__ = [
  'CONTINUOUS',
  'DISCRETE',
]


CONTINUOUS = 'continuous'
DISCRETE = 'discrete'

F_vmap_fx = 'F_vmap_fx'
F_vmap_fy = 'F_vmap_fy'
F_vmap_brentq_fx = 'F_vmap_brentq_fx'
F_vmap_brentq_fy = 'F_vmap_brentq_fy'
F_vmap_fp_aux = 'vmap_fixed_point_aux'
F_vmap_fp_opt = 'vmap_fixed_point_opt'
F_vmap_dfxdx = 'F_vmap_dfxdx'
F_fx = 'F_fx'
F_fy = 'F_fy'
F_fz = 'F_fz'
F_dfxdx = 'F_dfxdx'
F_dfxdy = 'F_dfxdy'
F_dfydx = 'F_dfydx'
F_dfydy = 'F_dfydy'
F_jacobian = 'F_jacobian'
F_vmap_jacobian = 'F_vmap_jacobian'
F_fixed_point = 'F_fixed_point'
F_fixed_point_aux = 'F_fixed_point_aux'
F_fixed_point_opt = 'F_fixed_point_opt'
F_fx_nullcline_by_opt = 'F_fx_nullcline_by_opt'
F_fy_nullcline_by_opt = 'F_fy_nullcline_by_opt'
F_x_in_all = 'F_x_in_all'
F_y_in_all = 'F_y_in_all'
F_x_by_y = 'F_x_by_y'
F_y_by_x = 'F_y_by_x'
F_y_convert = 'F_y_convert'
F_x_convert = 'F_x_convert'
F_int_x = 'F_int_x'
F_int_y = 'F_int_y'

x_by_y = 'x_by_y'
y_by_x = 'y_by_x'
y_by_x_in_fy = 'fy::y=f(x)'
y_by_x_in_fx = 'fx::y=f(x)'
x_by_y_in_fx = 'fx::x=f(y)'
x_by_y_in_fy = 'fy::x=f(y)'
F_y_by_x_in_fy = 'F[fy::y=f(x)]'
F_x_by_y_in_fy = 'F[fy::x=f(y)]'
F_y_by_x_in_fx = 'F[fx::y=f(x)]'
F_x_by_y_in_fx = 'F[fx::x=f(y)]'
fx_nullcline_points = 'fx_nullcline_points'
fy_nullcline_points = 'fy_nullcline_points'
sympy_failed = 'sympy_failed'
sympy_success = 'sympy_success'
sympy_escape = 'sympy_escape'
sympy_timeout = 'sympy_timeout'
fx_sign = 'fx_sign'
fy_sign = 'fy_sign'
par_eval_parallel = 'par_eval_parallel'
par_eval_iter = 'par_eval_iter'
prefix = '\t'

