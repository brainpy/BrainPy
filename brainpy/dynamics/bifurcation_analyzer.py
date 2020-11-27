
import brainpy as bp
from brainpy.dynamics import PhasePortraitAnalyzer1D
from brainpy.dynamics import PhasePortraitAnalyzer2D

import sympy
import brainpy.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


__all__ = [
    'BifucationAnalyzer',
]


def _list_sympy2str(sym_list):
    if isinstance(sym_list, (tuple, list)):
        return list(map(str, sym_list))
    return sym_list


_plt_plot_scheme = {
                    "stable node":
                                {"color":(18/255, 44/255, 52/255)},
                    "unstable node":
                                {"color":(204/255, 219/255, 187/255)},
                    "stable focus":
                                {"color":(69/255, 73/255, 85/255)},
                    "unstable focus":
                                {"color":(243/255, 239/255, 245/255)},
                    "saddle node":
                                {"color":(244/255, 159/255, 10/255)},
                    "saddle focus":
                                {"color":(184/255, 225/255, 255/255)},
                    "undetermined":
                                {"color":(214/255, 34/255, 70/255),
                                 "marker":"^"},
                    }


class BifucationAnalyzer(object):
    ''' A tool class for bifurcation analysis in 1/2D dynamical systems.
    
    The bifurcation analyzer is restricted to analyze the bifurcation
    relation between membrane potential and a given model parameter
    (codimension-1 case) or two model parameters (codimension-2 case).
    
    Externally injected current is also treated as a model parameter in
    this class, instead of a model state.
    '''
    def __init__(self, neuron, var_lim, plot_var, parameter, 
                    sub_dict=None, resolution=None):
        """Parameters

        neuron :  NeuType of BrainPy
        An abstract neuronal type defined in BrainPy.

        var_lim : dict
            A dictionary containing the range of each variable. 
            Format: {"variable A": [A_min, A_max], ...}

        plot_var : str
            A str specifying the free variables.

        parameter : dict
            A dictionary specifying all the parameters in the model and 
            their ranges.
            Format: {"param A": [A_min, A_max], ...}

        sub_dict : dict, optional
            A dictionary containing the freeze values for other non-free 
            parameters. Default is `None`.
            Format: {"variable A": A_value, ...}

        resolution: dict, optinal
            A dictionary specifying simulation granularity for each parameter. 
            Default is 30.
            Format: {"param A": A_resolution, ...}
        
        """
        self.neuro = neuron
        self.parameter = parameter
        self.p_var = plot_var
        self.var_lim = var_lim
        if sub_dict is None:
            sub_dict = {}
        self.sub_dict = sub_dict
        if resolution is None:
            resolution = {k:30 for k in self.parameter.keys()}
        self.resolution = resolution


        if len(parameter) == 1:
            self.analyzer = _BifurcationAnalyzerCo1(neuron, 
                                                    var_lim,
                                                    plot_var, 
                                                    parameter,
                                                    sub_dict,
                                                    resolution)
        elif len(parameter) == 2:
            self.analyzer = _BifurcationAnalyzerCo2(neuron, 
                                                    var_lim,
                                                    plot_var, 
                                                    parameter,
                                                    sub_dict,
                                                    resolution)
        else:
            raise RuntimeError("The number of parameters in bifurcation"
                "analysis cannot exceed 2.")


    def run(self):
        axis, data = self.analyzer.run(self.analyzer.dim)
        self.analyzer.show_results(axis, data)



class _CoDimAnalyzerBase(object):
    def __init__(self, neuron, var_lim, plot_var, parameter, sub_dict, resolution, dim):
        self.neuro = neuron
        self.p = list(parameter.keys())
        self.p_var = plot_var
        self.res = resolution
        self.var_lim = var_lim
        self.sub_dict = sub_dict
        self.varstr_list = _list_sympy2str(self._get_vars())
        self.dim = dim
        assert len(parameter) == dim, f"The number of parameters should be equal to the codimensionality."
        self.axis = [np.linspace(*parameter[self.p[i]], self.res[self.p[i]]) for i in range(dim)]


    def _get_vars(self):
        var_list = []
        for int_exp in self.neuro.integrators:
            diff_eq  = int_exp.diff_eq
            var_name = sympy.Symbol(int_exp.diff_eq.var_name, real=True)
            var_list.append(var_name)
        return var_list

    def run(self, n, p_values={}):
        data = {"unstable node":[], "stable node":[], 
                "unstable focus":[], "stable focus":[], 
                "saddle node": [], "saddle focus":[],
                "undetermined":[]}

        axis = [{"unstable node":[], "stable node":[], 
                 "unstable focus":[], "stable focus":[], 
                 "saddle node": [], "saddle focus":[],
                 "undetermined":[]}  for _ in range(self.dim)]

        if n >= 1:
            for pv in self.axis[n-1]:
                p_values.update({self.p[n-1]:pv})
                axis_, data_ = self.run(n-1, p_values)
                for k in data.keys():
                    for i in range(self.dim):
                        axis[i][k] += axis_[i][k]
                    data[k] += data_[k]
            return axis, data
        else:
            fp_list = self._find_fixed_point(p_values)
            for fp in fp_list:
                if fp[1] == "unstable node":
                    data["unstable node"].append(fp[0])
                    for i in range(self.dim):
                        axis[i]["unstable node"].append(p_values[self.p[i]])

                elif fp[1] == "stable node":
                    data["stable node"].append(fp[0])
                    for i in range(self.dim):
                        axis[i]["stable node"].append(p_values[self.p[i]])

                elif fp[1] == "unstable focus":
                    data["unstable focus"].append(fp[0])
                    for i in range(self.dim):
                        axis[i]["unstable focus"].append(p_values[self.p[i]])

                elif fp[1] == "stable focus":
                    data["stable focus"].append(fp[0])
                    for i in range(self.dim):
                        axis[i]["stable focus"].append(p_values[self.p[i]])

                else:
                    data["undetermined"].append(fp[0])
                    for i in range(self.dim):
                        axis[i]["undetermined"].append(p_values[self.p[i]])
                    print("undetermined type.")         # FIXME

            return axis, data


    def _find_fixed_point(self, sub_dict):
        if len(self.varstr_list) == 1:
            self.da = PhasePortraitAnalyzer1D(neuron=self.neuro, 
                                              plot_variables=self.varstr_list,
                                              param_update=sub_dict)
        elif len(self.varstr_list) == 2:
            # todo: need further test
            self.da = PhasePortraitAnalyzer2D(neuron=self.neuro, 
                                              plot_variables=self.varstr_list,
                                              param_update=sub_dict)

        return self.da.find_fixed_point(self.var_lim, sub_dict=sub_dict, 
                        suppress_print=True, suppress_plot=True)


    def show_results(self, axis, data):
        raise NotImplementedError



class _BifurcationAnalyzerCo1(_CoDimAnalyzerBase):
    def __init__(self, neuron, var_lim, plot_var, parameter, sub_dict, resolution):
        super(_BifurcationAnalyzerCo1, self).__init__(neuron=neuron,
                                                      var_lim=var_lim,
                                                      plot_var=plot_var,
                                                      parameter=parameter,
                                                      sub_dict=sub_dict,
                                                      resolution=resolution,
                                                      dim=1)


    def show_results(self, axis, data):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = axis[0]
        for k in data.keys():
            if len(data[k]) > 0:
                ax.scatter(x[k][0], data[k][0], **_plt_plot_scheme[k], label=k)
            if len(data[k]) > 1:
                ax.scatter(x[k][1:], data[k][1:], **_plt_plot_scheme[k])

        ax.set_ylabel(self.p_var)
        ax.set_xlabel(self.p)
        ax.legend()
        ax.grid(True)



class _BifurcationAnalyzerCo2(_CoDimAnalyzerBase):
    def __init__(self, neuron, var_lim, plot_var, parameter, sub_dict, resolution):
        super(_BifurcationAnalyzerCo2, self).__init__(neuron=neuron,
                                                      var_lim=var_lim,
                                                      plot_var=plot_var,
                                                      parameter=parameter,
                                                      sub_dict=sub_dict,
                                                      resolution=resolution,
                                                      dim=2)

    def show_results(self, axis, data):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y = axis[1], axis[0]

        for k in data.keys():
            if len(data[k]) > 0:
                ax.scatter(x[k][0], y[k][0], data[k][0], **_plt_plot_scheme[k], label=k)
            if len(data[k]) > 1:
                ax.scatter(x[k][1:], y[k][1:], data[k][1:], **_plt_plot_scheme[k])


        ax.set_xlabel(self.p[1])
        ax.set_ylabel(self.p[0])
        ax.set_zlabel(self.p_var)
        ax.grid(True)
        ax.legend()

