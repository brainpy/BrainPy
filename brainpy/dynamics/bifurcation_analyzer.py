# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import base
from . import utils
from .. import core
from .. import errors

__all__ = [
    'BifurcationAnalyzer',
    'Bifurcation1DAnalyzer',
    'Bifurcation2DAnalyzer',
]


class BifurcationAnalyzer(object):
    """A tool class for bifurcation analysis.
    
    The bifurcation analyzer is restricted to analyze the bifurcation
    relation between membrane potential and a given model parameter
    (co-dimension-1 case) or two model parameters (co-dimension-2 case).
    
    Externally injected current is also treated as a model parameter in
    this class, instead of a model state.

    Parameters
    ----------

    model :  NeuType
        An abstract neuronal type defined in BrainPy.

    """

    def __init__(self, model, target_pars, target_vars, fixed_vars=None, pars_update=None,
                 numerical_resolution=0.1, options=None):

        # check "model"
        if not isinstance(model, core.NeuType):
            raise errors.ModelUseError('Bifurcation analysis only support neuron type model.')
        self.model = model

        # check "target_pars"
        if not isinstance(target_pars, dict):
            raise errors.ModelUseError('"target_pars" must a dict with the format of: '
                                       '{"Parameter A": [A_min, A_max],'
                                       ' "Parameter B": [B_min, B_max]}')
        self.target_pars = target_pars
        if len(target_pars) > 2:
            raise errors.ModelUseError("The number of parameters in bifurcation"
                                       "analysis cannot exceed 2.")

        # check "fixed_vars"
        if fixed_vars is None:
            fixed_vars = dict()
        if not isinstance(fixed_vars, dict):
            raise errors.ModelUseError('"fixed_vars" must be a dict the format of: '
                                       '{"Variable A": A_value, "Variable B": B_value}')
        self.fixed_vars = fixed_vars

        # check "target_vars"
        if not isinstance(target_vars, dict):
            raise errors.ModelUseError('"target_vars" must a dict with the format of: '
                                       '{"Variable A": [A_min, A_max], "Variable B": [B_min, B_max]}')
        self.target_vars = target_vars

        # check "pars_update"
        if pars_update is None:
            pars_update = dict()
        if not isinstance(pars_update, dict):
            raise errors.ModelUseError('"pars_update" must be a dict the format of: '
                                       '{"Par A": A_value, "Par B": B_value}')
        for key in pars_update.keys():
            if key not in model.step_scopes:
                raise errors.ModelUseError(f'"{key}" is not a valid parameter in "{model.name}" model. ')
        self.pars_update = pars_update

        # bifurcation analysis
        if len(self.target_vars) == 1:
            self.analyzer = Bifurcation1DAnalyzer(model=model,
                                                  target_pars=target_pars,
                                                  target_vars=target_vars,
                                                  fixed_vars=fixed_vars,
                                                  pars_update=pars_update,
                                                  numerical_resolution=numerical_resolution,
                                                  options=options)

        elif len(self.target_vars) == 2:
            self.analyzer = Bifurcation2DAnalyzer(model=model,
                                                  target_pars=target_pars,
                                                  target_vars=target_vars,
                                                  fixed_vars=fixed_vars,
                                                  pars_update=pars_update,
                                                  numerical_resolution=numerical_resolution,
                                                  options=options)

        else:
            raise errors.ModelUseError(f'Cannot analyze three dimensional system: {self.target_vars}')

    def plot_bifurcation(self, *args, **kwargs):
        self.analyzer.plot_bifurcation(*args, **kwargs)


class Bifurcation1DAnalyzer(base.Base1DNeuronAnalyzer):
    """Bifurcation analysis of 1D system.

    Using this class, we can make co-dimension1 or co-dimension2 bifurcation analysis.
    """

    def __init__(self, model, target_pars, target_vars, fixed_vars=None,
                 pars_update=None, numerical_resolution=0.1, options=None):
        super(Bifurcation1DAnalyzer, self).__init__(model=model,
                                                    target_pars=target_pars,
                                                    target_vars=target_vars,
                                                    fixed_vars=fixed_vars,
                                                    pars_update=pars_update,
                                                    numerical_resolution=numerical_resolution,
                                                    options=options)

    def plot_bifurcation(self, show=False):
        f_fixed_point = self.get_f_fixed_point()
        f_dfdx = self.get_f_dfdx()

        if len(self.target_pars) == 1:
            container = {c: {'p': [], 'x': []} for c in utils.get_1d_classification()}

            # fixed point
            par_a = self.dpar_names[0]
            for p in self.resolutions[par_a]:
                xs = f_fixed_point(p)
                for x in xs:
                    dfdx = f_dfdx(x, p)
                    fp_type = utils.stability_analysis(dfdx)
                    container[fp_type]['p'].append(p)
                    container[fp_type]['x'].append(x)

            # visualization
            for fp_type, points in container.items():
                if len(points['x']):
                    plot_style = utils.plot_scheme[fp_type]
                    plt.plot(points['p'], points['x'], '.', **plot_style, label=fp_type)
            plt.xlabel(par_a)
            plt.ylabel(self.x_var)
            plt.legend()
            if show:
                plt.show()

        elif len(self.target_pars) == 2:
            container = {c: {'p0': [], 'p1': [], 'x': []} for c in utils.get_1d_classification()}

            # fixed point
            for p0 in self.resolutions[self.dpar_names[0]]:
                for p1 in self.resolutions[self.dpar_names[1]]:
                    xs = f_fixed_point(p0, p1)
                    for x in xs:
                        dfdx = f_dfdx(x, p0, p1)
                        fp_type = utils.stability_analysis(dfdx)
                        container[fp_type]['p0'].append(p0)
                        container[fp_type]['p1'].append(p1)
                        container[fp_type]['x'].append(x)

            # visualization
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for fp_type, points in container.items():
                if len(points['x']):
                    plot_style = utils.plot_scheme[fp_type]
                    xs = points['p0']
                    ys = points['p1']
                    zs = points['x']
                    ax.scatter(xs, ys, zs, **plot_style, label=fp_type)
            ax.set_xlabel(self.dpar_names[0])
            ax.set_ylabel(self.dpar_names[1])
            ax.set_zlabel(self.x_var)
            ax.grid(True)
            ax.legend()
            if show:
                plt.show()

        else:
            raise errors.ModelUseError(f'Cannot visualize co-dimension {len(self.target_pars)} '
                                       f'bifurcation.')


class Bifurcation2DAnalyzer(base.Base2DNeuronAnalyzer):
    """Bifurcation analysis of 2D system.

    Using this class, we can make co-dimension1 or co-dimension2 bifurcation analysis.
    """

    def __init__(self, model, target_pars, target_vars, fixed_vars=None,
                 pars_update=None, numerical_resolution=0.1, options=None):
        super(Bifurcation2DAnalyzer, self).__init__(model=model,
                                                    target_pars=target_pars,
                                                    target_vars=target_vars,
                                                    fixed_vars=fixed_vars,
                                                    pars_update=pars_update,
                                                    numerical_resolution=numerical_resolution,
                                                    options=options)

    def plot_bifurcation(self, plot_vars, show=False):
        # check "plot_vars"
        if isinstance(plot_vars, str):
            plot_vars = [plot_vars]
        if not isinstance(plot_vars, (tuple, list)):
            raise errors.ModelUseError('"plot_vars" must a tuple/list.')
        for var in plot_vars:
            if var in self.fixed_vars:
                raise errors.ModelUseError(f'"{var}" is defined in "fixed_vars", '
                                           f'cannot be used to plot.')
            if var not in self.target_vars:
                raise errors.ModelUseError(f'"{var}" is not a dynamical variable, '
                                           f'cannot be used to plot.')

        # functions
        f_fixed_point = self.get_f_fixed_point()
        f_jacobian = self.get_f_jacobian()

        # bifurcation analysis of co-dimension 1
        if len(self.target_pars) == 1:
            container = {c: {'p': [], self.x_var: [], self.y_var: []}
                         for c in utils.get_2d_classification()}

            # fixed point
            for p in self.resolutions[self.dpar_names[0]]:
                xs, ys = f_fixed_point(p)
                for x, y in zip(xs, ys):
                    dfdx = f_jacobian(x, y, p)
                    fp_type = utils.stability_analysis(dfdx)
                    container[fp_type]['p'].append(p)
                    container[fp_type][self.x_var].append(x)
                    container[fp_type][self.y_var].append(y)

            # visualization
            for var in plot_vars:
                plt.figure()
                for fp_type, points in container.items():
                    if len(points['p']):
                        plot_style = utils.plot_scheme[fp_type]
                        plt.plot(points['p'], points[var], '.', **plot_style, label=fp_type)
                plt.xlabel(self.dpar_names[0])
                plt.ylabel(var)
                plt.legend()
            if show:
                plt.show()

        # bifurcation analysis of co-dimension 2
        elif len(self.target_pars) == 2:
            container = {c: {'p0': [], 'p1': [], self.x_var: [], self.y_var: []}
                         for c in utils.get_2d_classification()}

            # fixed point
            for p1 in self.resolutions[self.dpar_names[0]]:
                for p2 in self.resolutions[self.dpar_names[1]]:
                    xs, ys = f_fixed_point(p1, p2)
                    for x, y in zip(xs, ys):
                        dfdx = f_jacobian(x, y, p1, p2)
                        fp_type = utils.stability_analysis(dfdx)
                        container[fp_type]['p0'].append(p1)
                        container[fp_type]['p1'].append(p2)
                        container[fp_type][self.x_var].append(x)
                        container[fp_type][self.y_var].append(y)

            # visualization
            for var in plot_vars:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for fp_type, points in container.items():
                    if len(points['p0']):
                        plot_style = utils.plot_scheme[fp_type]
                        xs = points['p0']
                        ys = points['p1']
                        zs = points[var]
                        ax.scatter(xs, ys, zs, **plot_style, label=fp_type)
                ax.set_xlabel(self.dpar_names[0])
                ax.set_ylabel(self.dpar_names[1])
                ax.set_zlabel(var)
                ax.legend()
            if show:
                plt.show()


class SlowFastBifurcation():
    pass


if __name__ == '__main__':
    Axes3D
