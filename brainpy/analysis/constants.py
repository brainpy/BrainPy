# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
__all__ = [
    'CONTINUOUS',
    'DISCRETE',

    'F_vmap_fx',
    'F_vmap_fy',
    'F_vmap_brentq_fx',
    'F_vmap_brentq_fy',
    'F_vmap_fp_aux',
    'F_vmap_fp_opt',
    'F_vmap_dfxdx',
    'F_fx',
    'F_fy',
    'F_fz',
    'F_dfxdx',
    'F_dfxdy',
    'F_dfydx',
    'F_dfydy',
    'F_jacobian',
    'F_vmap_jacobian',
    'F_fixed_point_aux',
    'F_fixed_point_opt',
    'F_x_by_y',
    'F_y_by_x',
    'F_y_convert',
    'F_x_convert',
    'F_int_x',
    'F_int_y',
    'x_by_y',
    'y_by_x',
    'y_by_x_in_fy',
    'y_by_x_in_fx',
    'x_by_y_in_fx',
    'x_by_y_in_fy',
    'F_y_by_x_in_fy',
    'F_x_by_y_in_fy',
    'F_y_by_x_in_fx',
    'F_x_by_y_in_fx',
    'fx_nullcline_points',
    'fy_nullcline_points',
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
F_fixed_point_aux = 'F_fixed_point_aux'
F_fixed_point_opt = 'F_fixed_point_opt'
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
prefix = '\t'
