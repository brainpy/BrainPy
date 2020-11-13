# -*- coding: utf-8 -*-

import numbers
import warnings

import matplotlib.pyplot as plt
import sympy
from scipy import optimize
from sympy import Derivative as D

from .. import numpy as np
from ..integration import sympy_tools

__all__ = [
    'PhasePortraitAnalyzer2D'
]


class PhasePortraitAnalyzer2D(object):
    r"""Analyzer for 2-dimensional dynamical systems. 

    This class can be used for vector field visualization, nullcline solving 
    and fixed point finding.

    Nullcines are obtained by algebraically solving the following equations:

    .. math::

        \dot{x} = f(x,y)
        \dot{y} = g(x,y)

    Note this class can only handle two-dimensional dynamical systems. For high
    dimensional dynamical systems, it's often desirable to reduce them
    to low dimensions for the benefits of visualization, by fixing certain 
    variables. This class can take a dynamical system with arbitrary
    number of dimensionalities as input, as long as enough contraints are 
    provided to reduce the system to two dimensions.


    Parameters
    -------------
    neuron : NeuType of BrainPy
        An abstract neuronal type defined in BrainPy.

    plot_variables : list of strs
        A list containing two str specifying the two free variables. The 
        first variable will become the y-axis, while the second the x-axis.
    """

    def __init__(self, neuron, plot_variables):
        self.neuro = neuron
        self.var_list = None
        self.eq_sympy_dict = None
        self.plot_variables = plot_variables
        self._get_sympy_equation()
        self.fig = plt.figure()
        y_var, x_var = self.plot_variables
        self.y_ind = list(map(str, self.var_list)).index(y_var)
        self.x_ind = list(map(str, self.var_list)).index(x_var)
        self.xlim = None
        self.ylim = None
        self.sub_dict = None

    def _get_sympy_equation(self):
        self.var_list = []
        self.eq_sympy_dict = {}
        for int_exp in self.neuro.integrators:
            var_name = sympy.Symbol(int_exp.diff_eq.var_name, real=True)
            eq_array = int_exp.diff_eq.get_f_expressions(substitute="all")
            params = int_exp.diff_eq.func_scope

            sub_params = {sympy.Symbol(k, real=True): v
                          for (k, v) in params.items() if isinstance(v, numbers.Number)}

            assert len(eq_array) == 1
            eq_sympy = sympy_tools.str2sympy(eq_array[0].code)
            eq_sympy = eq_sympy.subs(sub_params)
            self.eq_sympy_dict[var_name] = eq_sympy
            self.var_list.append(var_name)

    def plot_nullcline(self, ylim=None, xlim=None, sub_dict=None, inherit=False):
        """Plot the nullcline.

        Parameters
        ----------
        ylim : list, optional
            A list containing the range of free variable y, in the format of 
            [y_min, y_max]. Default is `None`.

        xlim : list, optional
            A list containing the range of free variable x, in the format of 
            [x_min, x_max]. Default is `None`.

        sub_dict : dict, optional
            A dictionary containing the freeze values for other non-free 
            parameters. Default is `None`.

        inherit : bool, optional 
            Whether to inherit settings from the last plot. If set to `True`,
            will ignore all the other parameters. Default is `False`.   
        """
        if sub_dict is None:
            sub_dict = {}

        if inherit:
            ylim = self.ylim
            xlim = self.xlim
            sub_dict = self.sub_dict
        else:
            self.ylim = ylim
            self.xlim = xlim
            self.sub_dict = sub_dict

        reverse_axis = [False, False]

        sub_dict = {sympy.Symbol(k, real=True): v for (k, v) in sub_dict.items()}
        _eq_sympy_dict = {k: v.subs(sub_dict) for (k, v) in self.eq_sympy_dict.items()}

        x_var = self.var_list[self.x_ind]
        y_var = self.var_list[self.y_ind]
        y_eq = _eq_sympy_dict[y_var]
        x_eq = _eq_sympy_dict[x_var]

        # Nullcline of the y variable
        x = np.linspace(*xlim, 10000)
        y = np.linspace(*ylim, 10000)


        try:
            sub_relation = sympy.solve(y_eq, y_var)
            nc1_list, x1_list, y1_list = [], [], []
            for i_sub in sub_relation:
                nc1_ = sympy.lambdify(x_var, i_sub, "numpy")(x)
                nc1_list.append(nc1_[np.bitwise_and(nc1_ < ylim[1], nc1_ > ylim[0])])
                x1_list.append(x[np.bitwise_and(nc1_ < ylim[1], nc1_ > ylim[0])])
                y1_list.append(y)

        except NotImplementedError:
            sub_relation = sympy.solve(y_eq, x_var)
            nc1_list, x1_list, y1_list = [], [], []
            for i_sub in sub_relation:
                nc1_ = sympy.lambdify(y_var, i_sub, "numpy")(y)
                nc1_list.append(nc1_[np.bitwise_and(nc1_ < xlim[1], nc1_ > xlim[0])])
                y1_list.append(x[np.bitwise_and(nc1_ < xlim[1], nc1_ > xlim[0])])
                x1_list.append(x)
            reverse_axis[0] = True

        # Nullcline of the x variable

        try:
            sub_relation = sympy.solve(x_eq, y_var)
            nc2_list, x2_list, y2_list = [], [], []
            for i_sub in sub_relation:
                nc2_ = sympy.lambdify(x_var, i_sub, "numpy")(x)
                nc2_list.append(nc2_[np.bitwise_and(nc2_ < ylim[1], nc2_ > ylim[0])])
                x2_list.append(x[np.bitwise_and(nc2_ < ylim[1], nc2_ > ylim[0])])
                y2_list.append(y)

        except NotImplementedError:
            sub_relation = sympy.solve(x_eq, x_var)
            nc2_list, x2_list, y2_list = [], [], []
            for i_sub in sub_relation:
                nc2_ = sympy.lambdify(y_var, i_sub, "numpy")(y)
                nc2_list.append(nc2_[np.bitwise_and(nc2_ < xlim[1], nc2_ > xlim[0])])
                y2.append(x[np.bitwise_and(nc2_ < xlim[1], nc2_ > xlim[0])])
                x2.append(x)
            reverse_axis = True

        nc = [nc1_list, nc2_list]
        xx = [x1_list, x2_list]
        yy = [y1_list, y2_list]

        for i in range(2):
            if reverse_axis[i]:
                nc[i], xx[i] = yy[i], nc[i]



        x_style = dict(color='lightcoral', alpha=.7, linewidth=4)
        y_style = dict(color='cornflowerblue', alpha=.7, linewidth=4)


        for i in range(len(nc[0])):
            label = self.plot_variables[0] + " nullcline" if i ==0 else None
            plt.plot(xx[0][i], nc[0][i], **y_style, label=label)
        for i in range(len(nc[1])):
            label = self.plot_variables[1] + " nullcline" if i ==0 else None
            plt.plot(xx[1][i], nc[1][i], **x_style, label=label)


        plt.xlabel(str(x_var))
        plt.ylabel(str(y_var))

        plt.legend()

        return

    def plot_vector_field(self, ylim=None, xlim=None, sub_dict=None,
                          res_x=50, res_y=50, inherit=False):
        """Plot the vector field.

        Parameter
        -----------
        ylim : list, optional
            A list containing the range of free variable y, in the format of 
            [y_min, y_max]. Default is `None`.

        xlim : list, optional
            A list containing the range of free variable x, in the format of 
            [x_min, x_max]. Default is `None`.

        sub_dict : dict, optional
            A dictionary containing the freeze values for other non-free 
            parameters. Default is `None`.

        res_x : int
            The resolution in x axis, i.e. the number of points to evaluate 
            in x direction. Default is `50`.

        res_y : int
            The resolution in y axis, i.e. the number of points to evaluate 
            in y direction. Default is `50`.

        inherit : bool, optional 
            Whether to inherit settings from the last plot. If set to `True`,
            will ignore all the other parameters. Default is `False`.       
        """

        if sub_dict is None:
            sub_dict = {}

        if inherit:
            ylim = self.ylim
            xlim = self.xlim
            sub_dict = self.sub_dict
        else:
            self.ylim = ylim
            self.xlim = xlim
            self.sub_dict = sub_dict

        x_vec = np.linspace(xlim[0], xlim[1], res_x)
        y_vec = np.linspace(ylim[0], ylim[1], res_y)
        X, Y = np.meshgrid(x_vec, y_vec)
        _X, _Y = X.reshape(-1, ), Y.reshape(-1, )

        sub_dict = {sympy.Symbol(k, real=True): v for (k, v) in sub_dict.items()}
        _eq_sympy_dict = {k: v.subs(sub_dict) for (k, v) in self.eq_sympy_dict.items()}
        _var_list = [self.var_list[self.y_ind], self.var_list[self.x_ind]]

        # evaluate dy
        f = sympy.lambdify(_var_list, _eq_sympy_dict[self.var_list[self.y_ind]], "numpy")
        dy = f(_Y, _X).reshape(Y.shape)

        # evaluate dx
        g = sympy.lambdify(_var_list, _eq_sympy_dict[self.var_list[self.x_ind]], "numpy")
        dx = g(_Y, _X).reshape(X.shape)

        speed = np.sqrt(dx ** 2 + dy ** 2)
        lw = 0.5 + 5.5 * speed / speed.max()
        ax = self.axes[0]
        ax.streamplot(X, Y, dx, dy, linewidth=lw, arrowsize=1.2, density=1, color='thistle')

        x_var = self.var_list[self.x_ind]
        y_var = self.var_list[self.y_ind]
        plt.xlabel(str(x_var))
        plt.ylabel(str(y_var))

        return dy, dx, Y, X

    def find_fixed_point(self, ylim=None, xlim=None, sub_dict=None, inherit=False):
        """
        Parameters
        ----------
        ylim : list, optional
            A list containing the range of free variable y, in the format of 
            [y_min, y_max]. Default is `None`.

        xlim : list, optional
            A list containing the range of free variable x, in the format of 
            [x_min, x_max]. Default is `None`.

        sub_dict : dict, optional
            A dictionary containing the freeze values for other non-free 
            parameters. Default is `None`.

        inherit : bool, optional 
            Whether to inherit settings from the last plot. If set to `True`,
            will ignore all the other parameters. Default is `False`.   
        """

        if sub_dict is None:
            sub_dict = {}

        if inherit:
            ylim = self.ylim
            xlim = self.xlim
            sub_dict = self.sub_dict
        else:
            self.ylim = ylim
            self.xlim = xlim
            self.sub_dict = sub_dict

        sub_dict = {sympy.Symbol(k, real=True): v for (k, v) in sub_dict.items()}
        _eq_sympy_dict = {k: v.subs(sub_dict) for (k, v) in self.eq_sympy_dict.items()}

        x_var = self.var_list[self.x_ind]
        y_var = self.var_list[self.y_ind]
        y_eq = _eq_sympy_dict[y_var]
        x_eq = _eq_sympy_dict[x_var]

        try:
            sub_relation = {y_var: sympy.solve(y_eq, y_var)[0]}
            f_eq = x_eq.subs(sub_relation)
            f_var = x_var
            x_pivot = True
        except NotImplementedError:
            try:
                sub_relation = {x_var: sympy.solve(x_eq, x_var)[0]}
                f_eq = y_eq.subs(sub_relation)
                f_var = y_var
                x_pivot = False
            except NotImplementedError:
                warnings.warn("Solving for fixed points failed. We cannot determine the "
                              "existence of fixed points algebraically. You can resort to numerical"
                              "methods by using ...(not implemented yet)")
                return

        # Solve for fixed points

        f_range = xlim if x_pivot else ylim
        f_optimizer = lambda x: sympy.lambdify(f_var, f_eq, "numpy")(x)
        fs = np.arange(*f_range) + 1e-6
        fs_len = len(fs)

        vals = f_optimizer(fs)
        fs_sign = np.sign(vals)

        fixed_points = []
        fl_sign = fs_sign[0]
        f_i = 1
        while f_i < fs_len and fl_sign == 0.:
            fixed_points.append(fs[f_i - 1])
            fl_sign = fs_sign[f_i]
            f_i += 1
        while f_i < fs_len:
            fr_sign = fs_sign[f_i]
            if fr_sign == 0.:
                fixed_points.append(fs[f_i])
                if f_i + 1 < fs_len:
                    fl_sign = fs_sign[f_i + 1]
                else:
                    break
                f_i += 2
            else:
                if not np.isnan(fr_sign) and fl_sign != fr_sign:
                    fixed_points.append(optimize.brentq(f_optimizer, fs[f_i - 1], fs[f_i]))
                fl_sign = fr_sign
                f_i += 1

        # determine fixed point types

        x_sol, y_sol = [], []
        for i in range(len(fixed_points)):
            if x_pivot:
                _y_sol = sub_relation[y_var].subs(x_var, fixed_points[i])
                if _y_sol >= ylim[0] and _y_sol <= ylim[1]:
                    x_sol.append(fixed_points[i])
                    y_sol.append(_y_sol)
            else:
                _x_sol = sub_relation[x_var].subs(y_var, fixed_points[i])
                if _x_sol >=xlim[0] and _x_sol <= xlim[1]:
                    y_sol.append(fixed_points[i])
                    x_sol.append(_x_sol)

        n_sol = len(x_sol)

        if n_sol == 0:
            print("No fixed point existed in this area. If you are in doubt, you can"
                  "resort to numerical methods by using ...(not implemented yet)")
            return

        jacob_mat_sympy = [[D(x_eq, x_var).doit(), D(x_eq, y_var).doit()],
                           [D(y_eq, x_var).doit(), D(y_eq, y_var).doit()]]

        for i in range(n_sol):
            jacob_mat = [[jacob_mat_sympy[0][0].subs({x_var: x_sol[i], y_var: y_sol[i]}),
                          jacob_mat_sympy[0][1].subs({x_var: x_sol[i], y_var: y_sol[i]})],
                         [jacob_mat_sympy[1][0].subs({x_var: x_sol[i], y_var: y_sol[i]}),
                          jacob_mat_sympy[1][1].subs({x_var: x_sol[i], y_var: y_sol[i]})]]

            if not (isinstance(jacob_mat[0][0], numbers.Number) and
                    isinstance(jacob_mat[0][1], numbers.Number) and
                    isinstance(jacob_mat[1][0], numbers.Number) and
                    isinstance(jacob_mat[1][1], numbers.Number)):
                raise RuntimeError("Undefined function found in Jacobian matrix."
                                   "It could be a result of insufficient contraints provided to reduce"
                                   "the dynamical system to 2D.")

            jacob_mat_np = np.array(jacob_mat)
            det = np.float64(jacob_mat_np[0, 0] * jacob_mat_np[1, 1] - jacob_mat_np[0, 1] * jacob_mat_np[1, 0])
            tr = np.float64(jacob_mat_np[0, 0] + jacob_mat_np[1, 1])

            eigenval = [(tr - np.sqrt(tr ** 2 - 4 * det + 0j)) / 2,
                        (tr + np.sqrt(tr ** 2 - 4 * det + 0j)) / 2]

            fp_info = lambda ind, fptype: print(f"Fixed point #{ind + 1} at {str(x_var)}={x_sol[i]}"
                                                f" and {str(y_var)}={y_sol[i]} is a/an {fptype}.")

            hollow = dict(markerfacecolor='white', markeredgecolor='black', markeredgewidth=2, markersize=20)
            solid = dict(color='black', markersize=20)

            # Judging types
            if np.all(np.isreal(eigenval)):
                if np.prod(eigenval) < 0:
                    fp_info(i, "saddle node")
                    plot_type = hollow
                elif eigenval[0] > 0:
                    fp_info(i, "unstable node")
                    plot_type = hollow
                else:
                    fp_info(i, "stable node")
                    plot_type = solid
            else:
                if np.all(np.real(eigenval) > 0):
                    fp_info(i, "unstable focus")
                    plot_type = hollow
                elif np.all(np.real(eigenval) < 0):
                    fp_info(i, "stable focus")
                    plot_type = solid
                else:
                    fp_info(i, "saddle focus")  # FIXME: undefined, need correction !!!
                    plot_type = hollow

            plt.plot(x_sol[i], y_sol[i], '.', **plot_type)
            plt.xlabel(str(x_var))
            plt.ylabel(str(y_var))


    def plot_trajectory(self, initial_states, input=None, dur=1000.):
        '''Plot the trajectory given initial states

        initial_states : dict
            A dict containing two items, with (key, value) pair specifying the initial values
            of the states (i.e. the `plot_variables` in the class constructor). If the `value`s in 
            the pair are lists of numbers, multiple corresponding trajectories will be plotted.

        input : float or array-like, optional
            The input provided to the neuron during simulation. Default is `None`.

        dur : float, optional
            A number specifying trajectory simulation duration. Default is 1000.
        '''

        # FIXME: Only 2D system is supported! Cannot be used to plot trajectories of reduced systems
        # as variables cannot be clamped during simulation !!!

        assert isinstance(initial_states, dict)

        x_str = str(self.var_list[self.x_ind])
        y_str = str(self.var_list[self.y_ind])

        x_val, y_val = initial_states[x_str], initial_states[y_str]
            
        if not(isinstance(x_val, (list, tuple)) and isinstance(y_val, (list, tuple))):
            assert isinstance(x_val, numbers.Number)
            assert isinstance(y_val, numbers.Number)

            x_val = [x_val]
            y_val = [y_val]


        else:

            assert isinstance(x_val,  (list, tuple))
            assert isinstance(y_val,  (list, tuple))
            assert len(x_val) == len(y_val)


        n_r = len(x_val)
        group = NeuGroup(self.neuro, geometry=n_r, monitors=[x_str, y_str])
        group.ST[x_str] = x_val
        group.ST[y_str] = y_val

        if input is None: input = 0 

        group.run(duration=dur, inputs=('ST.input', input))

        plt.plot(group.mon[x_str], group.mon[y_str], color="darkgoldenrod", linewidth=.7, alpha=.7)

        return





    @property
    def axes(self):
        return self.fig.axes
