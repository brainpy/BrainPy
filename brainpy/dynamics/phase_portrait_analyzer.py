import sys
sys.path.append("../BrainPy/")
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


def _convert2sympy_dict(d):
    if isinstance(d, dict):
        _d = {}
        for (k, v) in d.items():
            if not isinstance(k, sympy.Symbol):
                _d[sympy.Symbol(k, real=True)] = v
            else:
                _d[k] = v
        d = _d
    return d


class _PhasePortraitAnalyzerBase(object):
    def __init__(self, neuron, plot_variables):
        self.neuro = neuron
        self.var_list = None
        self.eq_sympy_dict = None
        self.plot_variables = plot_variables
        self._get_sympy_equation()
        self.fig = plt.figure()
        self.sub_dict = None
        self.var_ind = [list(map(str, self.var_list)).index(_var) 
                                        for _var in plot_variables]
        self.var_lim = None
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


    def plot_vector_field(self, var_lim=None, sub_dict=None, 
                            var_res=50, inherit=False):
        '''Plot the vector field.

        Parameter
        -----------
        var_lim: dict, optional
            todo

        sub_dict : dict, optional
            A dictionary containing the freeze values for other non-free 
            parameters. Default is `None`.

        var_res : int or dict, optional
            The vector field spatial resolution. Default is `50`.

        inherit : bool, optional 
            Whether to inherit settings from the last plot. If set to `True`,
            will ignore all the other parameters values. Default is `False`.       
        '''
        var_lim = _convert2sympy_dict(var_lim)

        if inherit:
            var_lim = self.var_lim
            sub_dict = self.sub_dict
        else:
            self.var_lim = var_lim
            self.sub_dict = sub_dict

        n_var = len(self.var_list)
        # _var_list = list(map(str, self.var_list))

        if isinstance(var_res, numbers.Number):
            var_res = {v:var_res for v in self.var_list}
        elif isinstance(var_res, dict):
            assert len(var_res.items()) == n_var
            var_res = _convert2sympy_dict(var_res)
        else:
            raise TypeError

        sub_dict = {sympy.Symbol(k, real=True):v for (k,v) in sub_dict.items()}
        _eq_sympy_dict = {k:v.subs(sub_dict) for (k, v) in self.eq_sympy_dict.items()}


        v_vec = {}

        for var in self.var_list:
            v_vec[var] = np.linspace(var_lim[var][0], var_lim[var][1], var_res[var])

        if n_var == 1:
            eval_points = [v_vec[self.var_list[0]]]
            eval_var = self.var_list[0]
            plot_points = eval_points
        elif n_var == 2:
            x_var = self.var_list[self.var_ind[1]]
            y_var = self.var_list[self.var_ind[0]]
            X, Y = np.meshgrid(v_vec[x_var], v_vec[y_var])  
            _X, _Y = X.reshape(-1,), Y.reshape(-1,) 
            eval_points = [_Y, _X]
            eval_var = [y_var, x_var]
            plot_points = [Y, X]
        else:
            raise NotImplementedError
    
        _shape = plot_points[0].shape

        dv = {}
        for var in self.var_list:
            fv = sympy.lambdify(eval_var, _eq_sympy_dict[var], "numpy")
            dv[var] = fv(*eval_points).reshape(_shape)

        self._plot_vector_field(dv, plot_points)


    def _plot_vector_field(self):
        raise NotImplementedError


    @property
    def axes(self):
        return self.fig.axes





class PhasePortraitAnalyzer1D(_PhasePortraitAnalyzerBase):
    def __init__(self, neuron, plot_variables):
        super(PhasePortraitAnalyzer1D, self).__init__(neuron, plot_variables)
        assert len(plot_variables) == 1
        self.x_var = self.var_list[0]


    def _plot_vector_field(self, dv, eval_points):
        dv_style = dict(color='lightcoral', alpha=.7, linewidth=4)
        x_style = dict(color='dimgrey', alpha=.7, linewidth=2)

        _x = eval_points[0]
        _y = np.linspace(-0.001, 0.001, _x.shape[0])

        plt.plot(_x, dv[self.x_var], **dv_style)
        plt.plot(_x, _y, **x_style)

        ax = self.axes[0]

        X, Y = np.meshgrid(_x, _y)  
        _X, _Y = X.reshape(-1,), Y.reshape(-1,) 

        dx = dv[self.x_var][None,:].repeat(_y.shape[0], 0)
        dy = np.ones(dx.shape) * dx.min() * 0.01

        ax.streamplot(X, Y, dx, dy, color='dimgrey')

        plt.xlabel(str(self.x_var))
        plt.ylabel(f"f'({str(self.x_var)})")


    def find_fixed_point(self, var_lim=None, sub_dict=None, inherit=False):
        """
        Parameters
        ----------
        var_lim: dict, optional
            todo

        sub_dict : dict, optional
            A dictionary containing the freeze values for other non-free 
            parameters. Default is `None`.

        inherit : bool, optional 
            Whether to inherit settings from the last plot. If set to `True`,
            will ignore all the other parameters. Default is `False`.   
        """
        var_lim = _convert2sympy_dict(var_lim)

        if inherit:
            var_lim = self.var_lim
            sub_dict = self.sub_dict
        else:
            self.var_lim = var_lim
            self.sub_dict = sub_dict

        sub_dict = {sympy.Symbol(k, real=True):v for (k,v) in sub_dict.items()}
        _eq_sympy_dict = {k:v.subs(sub_dict) for (k, v) in self.eq_sympy_dict.items()}

        x_eq = _eq_sympy_dict[self.x_var]


        # Solve for fixed points

        xlim = var_lim[self.x_var]
        f_var = self.x_var
        f_eq = x_eq
        f_range = xlim

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
        x_sol = fixed_points
        n_sol = len(x_sol)

        if n_sol == 0:
            print("No fixed point existed in this area. If you are in doubt, you can"
                "resort to numerical methods by using ...(not implemented yet)")
            return

        hollow = dict(markerfacecolor='white', markeredgecolor='black', markeredgewidth=2, markersize=20)
        solid = dict(color='black', markersize=20)

        fp_info = lambda ind, fptype: print(f"Fixed point #{ind+1} at {str(self.x_var)}={x_sol[i]}"
                                                f" is a/an {fptype}.")

        ddx = D(x_eq, self.x_var).doit()

        for i in range(n_sol):
            ddx_ = ddx.subs({self.x_var:x_sol[i]})
            if ddx_ < 0:
                plot_type = solid
                plt.plot(x_sol[i], 0, '.', **plot_type)
                fp_info(i, "stable node")
            elif ddx_ > 0:         
                plot_type = hollow
                plt.plot(x_sol[i], 0, '.', **plot_type)
                fp_info(i, "unstable node")

            else:           # todo: take higher-order derivatives
                plot_type = hollow
                plt.plot(x_sol[i], 0, '.', **plot_type)
                fp_info(i, "undetermined type because higher-order derivative need to be calculated. "
                    "This will be fixed in a future update.")



class PhasePortraitAnalyzer2D(_PhasePortraitAnalyzerBase):
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
        super(PhasePortraitAnalyzer2D, self).__init__(neuron, plot_variables)
        assert len(plot_variables) == 2
        self.y_var = self.var_list[self.var_ind[0]]
        self.x_var = self.var_list[self.var_ind[1]]


    def _plot_vector_field(self, dv, pts):
        dy, dx = dv[self.y_var], dv[self.x_var]
        Y, X = pts
        speed = np.sqrt(dx**2 + dy**2)
        lw = 0.5 + 5.5*speed / speed.max()
        ax = self.axes[0]
        ax.streamplot(X, Y, dx, dy, linewidth=lw, arrowsize=1.2, density=1, color='thistle')

        plt.xlabel(str(self.x_var))
        plt.ylabel(str(self.y_var))



    def plot_nullcline(self, var_lim=None, sub_dict=None, inherit=False):
        """Plot the nullcline.

        Parameters
        ----------
        var_lim : dict, optional
            todo

        sub_dict : dict, optional
            A dictionary containing the freeze values for other non-free 
            parameters. Default is `None`.

        inherit : bool, optional 
            Whether to inherit settings from the last plot. If set to `True`,
            will ignore all the other parameters. Default is `False`.   
        """
        var_lim = _convert2sympy_dict(var_lim)

        if sub_dict is None:
            sub_dict = {}

        if inherit:
            var_lim = self.var_lim
            sub_dict = self.sub_dict
        else:
            self.var_lim = var_lim
            self.sub_dict = sub_dict

        reverse_axis = [False, False]

        sub_dict = {sympy.Symbol(k, real=True): v for (k, v) in sub_dict.items()}
        _eq_sympy_dict = {k: v.subs(sub_dict) for (k, v) in self.eq_sympy_dict.items()}

        y_eq = _eq_sympy_dict[self.y_var]
        x_eq = _eq_sympy_dict[self.x_var]
        ylim = var_lim[self.y_var]
        xlim = var_lim[self.x_var]

        # Nullcline of the y variable
        x = np.linspace(*xlim, 10000)
        y = np.linspace(*ylim, 10000)

        try:
            sub_relation = sympy.solve(y_eq, self.y_var)
            nc1_list, x1_list, y1_list = [], [], []
            for i_sub in sub_relation:
                nc1_ = sympy.lambdify(self.x_var, i_sub, "numpy")(x)
                nc1_list.append(nc1_[np.bitwise_and(nc1_ < ylim[1], nc1_ > ylim[0])])
                x1_list.append(x[np.bitwise_and(nc1_ < ylim[1], nc1_ > ylim[0])])
                y1_list.append(y)

        except NotImplementedError:
            sub_relation = sympy.solve(y_eq, self.x_var)
            nc1_list, x1_list, y1_list = [], [], []
            for i_sub in sub_relation:
                nc1_ = sympy.lambdify(self.y_var, i_sub, "numpy")(y)
                nc1_list.append(nc1_[np.bitwise_and(nc1_ < xlim[1], nc1_ > xlim[0])])
                y1_list.append(x[np.bitwise_and(nc1_ < xlim[1], nc1_ > xlim[0])])
                x1_list.append(x)
            reverse_axis[0] = True

        # Nullcline of the x variable

        try:
            sub_relation = sympy.solve(x_eq, self.y_var)
            nc2_list, x2_list, y2_list = [], [], []
            for i_sub in sub_relation:
                nc2_ = sympy.lambdify(self.x_var, i_sub, "numpy")(x)
                nc2_list.append(nc2_[np.bitwise_and(nc2_ < ylim[1], nc2_ > ylim[0])])
                x2_list.append(x[np.bitwise_and(nc2_ < ylim[1], nc2_ > ylim[0])])
                y2_list.append(y)

        except NotImplementedError:
            sub_relation = sympy.solve(x_eq, self.x_var)
            nc2_list, x2_list, y2_list = [], [], []
            for i_sub in sub_relation:
                nc2_ = sympy.lambdify(self.y_var, i_sub, "numpy")(y)
                nc2_list.append(nc2_[np.bitwise_and(nc2_ < xlim[1], nc2_ > xlim[0])])
                y2_list.append(x[np.bitwise_and(nc2_ < xlim[1], nc2_ > xlim[0])])
                x2_list.append(x)
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
            label = self.plot_variables[0] + " nullcline" if i == 0 else None
            plt.plot(xx[0][i], nc[0][i], **y_style, label=label)
        for i in range(len(nc[1])):
            label = self.plot_variables[1] + " nullcline" if i == 0 else None
            plt.plot(xx[1][i], nc[1][i], **x_style, label=label)

        plt.xlabel(str(self.x_var))
        plt.ylabel(str(self.y_var))

        plt.legend()

        return 
    


    def find_fixed_point(self, var_lim=None, sub_dict=None, inherit=False):
        """
        Parameters
        ----------
        var_lim: dict, optional
            todo

        sub_dict : dict, optional
            A dictionary containing the freeze values for other non-free 
            parameters. Default is `None`.

        inherit : bool, optional 
            Whether to inherit settings from the last plot. If set to `True`,
            will ignore all the other parameters. Default is `False`.   
        """

        var_lim = _convert2sympy_dict(var_lim)

        if inherit:
            var_lim = self.var_lim
            sub_dict = self.sub_dict
        else:
            self.var_lim = var_lim
            self.sub_dict = sub_dict

        sub_dict = {sympy.Symbol(k, real=True):v for (k,v) in sub_dict.items()}
        _eq_sympy_dict = {k:v.subs(sub_dict) for (k, v) in self.eq_sympy_dict.items()}

        y_eq = _eq_sympy_dict[self.y_var]
        x_eq = _eq_sympy_dict[self.x_var]

        try:
            sub_relation = {self.y_var: sympy.solve(y_eq, self.y_var)[0]}
            f_eq = x_eq.subs(sub_relation)
            f_var = self.x_var
            x_pivot = True
        except NotImplementedError:
            try:
                sub_relation = {self.x_var: sympy.solve(x_eq, self.x_var)[0]}
                f_eq = y_eq.subs(sub_relation)
                f_var = self.y_var
                x_pivot = False
            except NotImplementedError: 
                warnings.warn("Solving for fixed points failed. We cannot determine the "
                "existence of fixed points algebraically. You can resort to numerical"
                "methods by using ...(not implemented yet)")
                return

        # Solve for fixed points

        xlim = var_lim[self.x_var]
        ylim = var_lim[self.y_var]

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
                y_sol.append(sub_relation[self.y_var].subs(self.x_var, fixed_points[i]))
            else:
                y_sol.append(fixed_points[i])
                x_sol.append(sub_relation[self.x_var].subs(self.y_var, fixed_points[i]))

        n_sol = len(x_sol)

        if n_sol == 0:
            print("No fixed point existed in this area. If you are in doubt, you can"
                "resort to numerical methods by using ...(not implemented yet)")
            return

        jacob_mat_sympy = [[D(x_eq, self.x_var).doit(), D(x_eq, self.y_var).doit()], 
                           [D(y_eq, self.x_var).doit(), D(y_eq, self.y_var).doit()]]

        for i in range(n_sol):
            jacob_mat = [[jacob_mat_sympy[0][0].subs({self.x_var:x_sol[i], self.y_var:y_sol[i]}),
                            jacob_mat_sympy[0][1].subs({self.x_var:x_sol[i], self.y_var:y_sol[i]})],
                         [jacob_mat_sympy[1][0].subs({self.x_var:x_sol[i], self.y_var:y_sol[i]}),
                            jacob_mat_sympy[1][1].subs({self.x_var:x_sol[i], self.y_var:y_sol[i]})]]

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

            fp_info = lambda ind, fptype: print(f"Fixed point #{ind+1} at {str(self.x_var)}={x_sol[i]}"
                                                f" and {str(self.y_var)}={y_sol[i]} is a/an {fptype}.")


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

        plt.xlabel(str(self.x_var))
        plt.ylabel(str(self.y_var))

        return









