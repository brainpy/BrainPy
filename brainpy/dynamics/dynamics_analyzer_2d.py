import matplotlib.pyplot as plt

import brainpy as nb
import brainpy.numpy as np
import sympy
import numbers
import warnings

from brainpy.integration import sympy_tools
from scipy import optimize
from sympy import Derivative as D

nb.profile.set_backend('numba')
nb.profile.set_method('euler')
nb.profile.set_dt(0.02)
nb.profile.merge_integral = False


class DynamicsAnalyzer2D(object):
    r'''Analyzer for 2-dimensional dynamical systems. 

    This class can be used for vector field visualization, nullcline solving 
    and fixed point finding.

    Nullcines are obtrained by algebraically solving the following equations:

    .. math::

        \dot{x} = f(x,y)
        \dot{y} = g(x,y)

    Note this class can only handle two-dimensional dynamical system. For high
    dimensional dynamical systems, it's often desirable to reduce them
    to low dimensions for the benefits of visualization, by fixing certain 
    variables. This class can take a dynamical system with arbitrary
    number of dimensionalities as input, as long as enough contraints are 
    provided to reduce the system to two dimensions.


    Parameters
    -------------
    neuro : NeuType of BrainPy
        An abstract neuronal type defined in BrainPy.

    plot_variables : list of str
        A list containing two str specifying the two free variables. The 
        first variable will become the y-axis, while the second the x-axis.
    '''
    def __init__(self, neuro, plot_variables):
        self.neuro = neuro
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
            diff_eq  = int_exp.diff_eq
            var_name = sympy.Symbol(int_exp.diff_eq.var_name, real=True)
            eq_array = int_exp.diff_eq.get_f_expressions(substitute="all")
            params   = int_exp.diff_eq.func_scope

            sub_params = {sympy.Symbol(k, real=True):v 
                for (k,v) in params.items() if isinstance(v, numbers.Number)}

            assert len(eq_array) == 1
            eq_sympy = sympy_tools.str2sympy(eq_array[0].code)
            eq_sympy = eq_sympy.subs(sub_params)
            self.eq_sympy_dict[var_name] = eq_sympy
            self.var_list.append(var_name)

    def plot_nullcline(self, ylim=None, xlim=None, sub_dict=None, inherit=False):
        '''Plot the nullcline.

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
            will ignore all the other parameters values. Default is `False`.   
        '''

        if inherit:
            ylim = self.ylim
            xlim = self.xlim
            sub_dict = self.sub_dict
        else:
            self.ylim = ylim
            self.xlim = xlim
            self.sub_dict = sub_dict

        reverse_axis = [False, False]

        sub_dict = {sympy.Symbol(k, real=True):v for (k,v) in sub_dict.items()}
        _eq_sympy_dict = {k:v.subs(sub_dict) for (k, v) in self.eq_sympy_dict.items()}

        x_var = self.var_list[self.x_ind]
        y_var = self.var_list[self.y_ind]
        y_eq = _eq_sympy_dict[y_var]
        x_eq = _eq_sympy_dict[x_var]

        # Nullcline of the 1st variable
        x = np.linspace(*xlim, 100)
        y = np.linspace(*ylim, 100)

        try:
            sub_relation = sympy.solve(y_eq, y_var)
            nc1_ = sympy.lambdify(x_var, sub_relation, "numpy")(x)[0]
            nc1 = nc1_[np.bitwise_and(nc1_<ylim[1], nc1_>ylim[0])]
            x1 = x[np.bitwise_and(nc1_<ylim[1], nc1_>ylim[0])]
            y1 = y
        except NotImplementedError:
            sub_relation = sympy.solve(y_eq, x_var)
            nc1_ = sympy.lambdify(y_var, sub_relation, "numpy")(y)[0]
            nc1 = nc1_[np.bitwise_and(nc1_<xlim[1], nc1_>xlim[0])]
            y1 = x[np.bitwise_and(nc1_<xlim[1], nc1_>xlim[0])]
            x1 = x
            reverse_axis[0] = True

        # Nullcline of the 2nd variable
        try:
            sub_relation = sympy.solve(x_eq, y_var)
            nc2_ = sympy.lambdify(x_var, sub_relation, "numpy")(x)[0]
            nc2 = nc2_[np.bitwise_and(nc2_<ylim[1], nc2_>ylim[0])]
            x2 = x[np.bitwise_and(nc2_<ylim[1], nc2_>ylim[0])]
            y2 = y
        except NotImplementedError:
            sub_relation = sympy.solve(x_eq, x_var)
            nc2_ = sympy.lambdify(y_var, sub_relation, "numpy")(y)[0]
            nc2 = nc2_[np.bitwise_and(nc2_<xlim[1], nc2_>xlim[0])]
            y2 = x[np.bitwise_and(nc2_<xlim[1], nc2_>xlim[0])]
            x2 = x
            reverse_axis = True

        nc = [nc1, nc2]
        xx = [x1, x2]
        yy = [y1, y2]

        for i in range(2):
            if reverse_axis[i]:
                nc[i], xx[i] = yy[i], nc[i]

        for i in range(2):
            plt.plot(xx[i], nc[i])

        plt.xlabel(str(x_var))
        plt.ylabel(str(y_var))

        plt.legend([f"{self.plot_variables[i]} nullcline" for i in range(2)])

        return 

    def plot_vector_field(self, ylim=None, xlim=None, sub_dict=None, 
                            res_x=50, res_y=50, inherit=False):
        '''Plot the vector field.

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
            will ignore all the other parameters values. Default is `False`.       
        '''

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
        _X, _Y = X.reshape(-1,), Y.reshape(-1,)

        sub_dict = {sympy.Symbol(k, real=True):v for (k,v) in sub_dict.items()}
        _eq_sympy_dict = {k:v.subs(sub_dict) for (k, v) in self.eq_sympy_dict.items()}
        _var_list = [self.var_list[self.y_ind], self.var_list[self.x_ind]]

        # evaluate dy
        f = sympy.lambdify(_var_list, _eq_sympy_dict[self.var_list[self.y_ind]], "numpy")
        dy = f(_Y, _X).reshape(Y.shape)

        #evaluate dx
        g = sympy.lambdify(_var_list, _eq_sympy_dict[self.var_list[self.x_ind]], "numpy")
        dx = g(_Y, _X).reshape(X.shape)


        speed = np.sqrt(dx**2 + dy**2)
        lw = 0.5 + 5.5*speed / speed.max()
        ax = self.axes[0]
        ax.streamplot(X, Y, dx, dy, linewidth=lw, arrowsize=1.2, density=1, color='thistle')

        x_var = self.var_list[self.x_ind]
        y_var = self.var_list[self.y_ind]
        plt.xlabel(str(x_var))
        plt.ylabel(str(y_var))

        return dy, dx, Y, X


    def find_fixed_point(self, ylim=None, xlim=None, sub_dict=None, inherit=False):
        '''
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
            will ignore all the other parameters values. Default is `False`.   
        '''

        if inherit:
            ylim = self.ylim
            xlim = self.xlim
            sub_dict = self.sub_dict
        else:
            self.ylim = ylim
            self.xlim = xlim
            self.sub_dict = sub_dict

        sub_dict = {sympy.Symbol(k, real=True):v for (k,v) in sub_dict.items()}
        _eq_sympy_dict = {k:v.subs(sub_dict) for (k, v) in self.eq_sympy_dict.items()}

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
            fixed_points.append(fs[f_i-1])
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
                    fixed_points.append(optimize.brentq(f_optimizer, fs[f_i-1], fs[f_i]))
                fl_sign = fr_sign
                f_i += 1

        # determine fixed point types

        x_sol, y_sol = [], []
        for i in range(len(fixed_points)):
            if x_pivot:
                x_sol.append(fixed_points[i])
                y_sol.append(sub_relation[y_var].subs(x_var, fixed_points[i]))
            else:
                y_sol.append(fixed_points[i])
                x_sol.append(sub_relation[x_var].subs(y_var, fixed_points[i]))

        n_sol = len(x_sol)

        if n_sol == 0:
            print("No fixed point existed in this area. If you are in doubt, you can"
                "resort to numerical methods by using ...(not implemented yet)")
            return

        jacob_mat_sympy = [[D(x_eq, x_var).doit(), D(x_eq, y_var).doit()], 
                           [D(y_eq, x_var).doit(), D(y_eq, y_var).doit()]]

        for i in range(n_sol):
            jacob_mat = [[jacob_mat_sympy[0][0].subs({x_var:x_sol[i], y_var:y_sol[i]}),
                            jacob_mat_sympy[0][1].subs({x_var:x_sol[i], y_var:y_sol[i]})],
                         [jacob_mat_sympy[1][0].subs({x_var:x_sol[i], y_var:y_sol[i]}),
                            jacob_mat_sympy[1][1].subs({x_var:x_sol[i], y_var:y_sol[i]})]]

            if not (isinstance(jacob_mat[0][0], numbers.Number) and \
                    isinstance(jacob_mat[0][1], numbers.Number) and \
                    isinstance(jacob_mat[1][0], numbers.Number) and \
                    isinstance(jacob_mat[1][1], numbers.Number)):
                raise RuntimeError("Undefined function found in Jacobian matrix."
                "It could be a result of insufficient contraints provided to reduce"
                "the dynamical system to 2D.")

            jacob_mat_np = np.array(jacob_mat)
            det = np.float64(jacob_mat_np[0,0]*jacob_mat_np[1,1]-jacob_mat_np[0,1]*jacob_mat_np[1,0])
            tr = np.float64(jacob_mat_np[0,0]+jacob_mat_np[1,1])

            eigenval = [(tr-np.sqrt(tr**2-4*det+0j))/2,
                     (tr+np.sqrt(tr**2-4*det+0j))/2]

            fp_info = lambda ind, fptype: print(f"Fixed point #{ind+1} at {str(x_var)}={x_sol[i]}"
                                                f" and {str(y_var)}={y_sol[i]} is a/an {fptype}.")


            hollow = dict(markerfacecolor='white', markeredgecolor='black', markeredgewidth=2, markersize=20)
            solid = dict(color='black', markersize=20)

            # Judging types
            if np.all(np.isreal(eigenval)):
                if np.prod(eigenval)<0:
                    fp_info(i, "saddle node")
                    plot_type = hollow
                elif eigenval[0]>0:
                    fp_info(i, "unstable node")
                    plot_type = hollow
                else:
                    fp_info(i, "stable node")
                    plot_type = solid
            else:
                if np.all(np.real(eigenval)>0):
                    fp_info(i, "unstable focus")
                    plot_type = hollow
                elif np.all(np.real(eigenval)<0):
                    fp_info(i, "stable focus")
                    plot_type = solid
                else:
                    fp_info(i, "saddle focus")     # FIXME: undefined, need correction !!!
                    plot_type = hollow

            plt.plot(x_sol[i], y_sol[i], '.', **plot_type)
            plt.xlabel(str(x_var))
            plt.ylabel(str(y_var))

        return


    @property
    def axes(self):
        return self.fig.axes
    



