# -*- coding: utf-8 -*-

from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import base
from . import utils
from .. import core
from .. import errors
from .. import profile

__all__ = [
    'Bifurcation',
    '_Bifurcation1D',
    '_Bifurcation2D',

    'FastSlowBifurcation',
    '_FastSlow1D',
    '_FastSlow2D',
]


class Bifurcation(object):
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
            self.analyzer = _Bifurcation1D(model=model,
                                           target_pars=target_pars,
                                           target_vars=target_vars,
                                           fixed_vars=fixed_vars,
                                           pars_update=pars_update,
                                           numerical_resolution=numerical_resolution,
                                           options=options)

        elif len(self.target_vars) == 2:
            self.analyzer = _Bifurcation2D(model=model,
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


class _Bifurcation1D(base.Base1DNeuronAnalyzer):
    """Bifurcation analysis of 1D system.

    Using this class, we can make co-dimension1 or co-dimension2 bifurcation analysis.
    """

    def __init__(self, model, target_pars, target_vars, fixed_vars=None,
                 pars_update=None, numerical_resolution=0.1, options=None):
        super(_Bifurcation1D, self).__init__(model=model,
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
            plt.figure(self.x_var)
            for fp_type, points in container.items():
                if len(points['x']):
                    plot_style = utils.plot_scheme[fp_type]
                    plt.plot(points['p'], points['x'], '.', **plot_style, label=fp_type)
            plt.xlabel(par_a)
            plt.ylabel(self.x_var)

            # scale = (self.options.lim_scale - 1) / 2
            # plt.xlim(*utils.rescale(self.target_pars[self.dpar_names[0]], scale=scale))
            # plt.ylim(*utils.rescale(self.target_vars[self.x_var], scale=scale))

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
            fig = plt.figure(self.x_var)
            ax = fig.gca(projection='3d')
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

            # scale = (self.options.lim_scale - 1) / 2
            # ax.set_xlim(*utils.rescale(self.target_pars[self.dpar_names[0]], scale=scale))
            # ax.set_ylim(*utils.rescale(self.target_pars[self.dpar_names[1]], scale=scale))
            # ax.set_zlim(*utils.rescale(self.target_vars[self.x_var], scale=scale))

            ax.grid(True)
            ax.legend()
            if show:
                plt.show()

        else:
            raise errors.ModelUseError(f'Cannot visualize co-dimension {len(self.target_pars)} '
                                       f'bifurcation.')


class _Bifurcation2D(base.Base2DNeuronAnalyzer):
    """Bifurcation analysis of 2D system.

    Using this class, we can make co-dimension1 or co-dimension2 bifurcation analysis.
    """

    def __init__(self, model, target_pars, target_vars, fixed_vars=None,
                 pars_update=None, numerical_resolution=0.1, options=None):
        super(_Bifurcation2D, self).__init__(model=model,
                                             target_pars=target_pars,
                                             target_vars=target_vars,
                                             fixed_vars=fixed_vars,
                                             pars_update=pars_update,
                                             numerical_resolution=numerical_resolution,
                                             options=options)

    def plot_bifurcation(self, show=False):
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
            for var in self.dvar_names:
                plt.figure(var)
                for fp_type, points in container.items():
                    if len(points['p']):
                        plot_style = utils.plot_scheme[fp_type]
                        plt.plot(points['p'], points[var], '.', **plot_style, label=fp_type)
                plt.xlabel(self.dpar_names[0])
                plt.ylabel(var)

                # scale = (self.options.lim_scale - 1) / 2
                # plt.xlim(*utils.rescale(self.target_pars[self.dpar_names[0]], scale=scale))
                # plt.ylim(*utils.rescale(self.target_vars[var], scale=scale))

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
            for var in self.dvar_names:
                fig = plt.figure(var)
                ax = fig.gca(projection='3d')
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

                # scale = (self.options.lim_scale - 1) / 2
                # ax.set_xlim(*utils.rescale(self.target_pars[self.dpar_names[0]], scale=scale))
                # ax.set_ylim(*utils.rescale(self.target_pars[self.dpar_names[1]], scale=scale))
                # ax.set_zlim(*utils.rescale(self.target_vars[var], scale=scale))

                ax.grid(True)
                ax.legend()
            if show:
                plt.show()


class FastSlowBifurcation(object):
    """Fast slow analysis analysis proposed by John Rinzel [1]_.

    (J Rinzel, 1985) proposed that in a fast-slow dynamical system,
    we can treat the slow variables as the bifurcation parameters, and
    then study how the different value of slow variables affect the
    bifurcation of the fast sub-system.


    References
    ----------

    .. [1] Rinzel, John. "Bursting oscillations in an excitable
           membrane model." In Ordinary and partial differential
           equations, pp. 304-316. Springer, Berlin, Heidelberg, 1985.

    """

    def __init__(self, model, fast_vars, slow_vars, fixed_vars=None,
                 pars_update=None, numerical_resolution=0.1, options=None):
        # check "model"
        if not isinstance(model, core.NeuType):
            raise errors.ModelUseError('FastSlowBifurcation only support neuron type model.')
        self.model = model

        # check "fast_vars"
        if not isinstance(fast_vars, dict):
            raise errors.ModelUseError('"fast_vars" must a dict with the format of: '
                                       '{"Var A": [A_min, A_max],'
                                       ' "Var B": [B_min, B_max]}')
        self.fast_vars = fast_vars
        if len(fast_vars) > 2:
            raise errors.ModelUseError("FastSlowBifurcation can only analyze the system with less "
                                       "than two-variable fast subsystem.")

        # check "slow_vars"
        if not isinstance(slow_vars, dict):
            raise errors.ModelUseError('"slow_vars" must a dict with the format of: '
                                       '{"Variable A": [A_min, A_max], '
                                       '"Variable B": [B_min, B_max]}')
        self.slow_vars = slow_vars
        if len(slow_vars) > 2:
            raise errors.ModelUseError("FastSlowBifurcation can only analyze the system with less "
                                       "than two-variable slow subsystem.")

        # check "fixed_vars"
        if fixed_vars is None:
            fixed_vars = dict()
        if not isinstance(fixed_vars, dict):
            raise errors.ModelUseError('"fixed_vars" must be a dict the format of: '
                                       '{"Variable A": A_value, "Variable B": B_value}')
        self.fixed_vars = fixed_vars

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
        if len(self.fast_vars) == 1:
            self.analyzer = _FastSlow1D(model=model,
                                        fast_vars=fast_vars,
                                        slow_vars=slow_vars,
                                        fixed_vars=fixed_vars,
                                        pars_update=pars_update,
                                        numerical_resolution=numerical_resolution,
                                        options=options)

        elif len(self.fast_vars) == 2:
            self.analyzer = _FastSlow2D(model=model,
                                        fast_vars=fast_vars,
                                        slow_vars=slow_vars,
                                        fixed_vars=fixed_vars,
                                        pars_update=pars_update,
                                        numerical_resolution=numerical_resolution,
                                        options=options)

        else:
            raise errors.ModelUseError(f'Cannot analyze {len(fast_vars)} dimensional fast system.')

    def plot_bifurcation(self, *args, **kwargs):
        self.analyzer.plot_bifurcation(*args, **kwargs)

    def plot_trajectory(self, *args, **kwargs):
        self.analyzer.plot_trajectory(*args, **kwargs)


class _FastSlowTrajectory(object):
    def __init__(self, model, fast_vars, slow_vars, fixed_vars=None,
                 pars_update=None, **kwargs):
        self.model = model
        self.fast_vars = fast_vars
        self.slow_vars = slow_vars
        self.fixed_vars = fixed_vars
        self.pars_update = pars_update
        options = kwargs.get('options', dict())
        if options is None:
            options = dict()
        self.lim_scale = options.get('lim_scale', 1.05)

        if isinstance(fast_vars, OrderedDict):
            self.fast_var_names = list(fast_vars.keys())
        else:
            self.fast_var_names = list(sorted(fast_vars.keys()))

        if isinstance(slow_vars, OrderedDict):
            self.slow_var_names = list(slow_vars.keys())
        else:
            self.slow_var_names = list(sorted(slow_vars.keys()))

        # cannot update dynamical parameters
        all_vars = self.fast_var_names + self.slow_var_names
        self.traj_group = core.NeuGroup(model,
                                        geometry=1,
                                        monitors=all_vars,
                                        pars_update=pars_update)
        self.traj_group.runner = core.TrajectoryRunner(self.traj_group,
                                                       target_vars=all_vars,
                                                       fixed_vars=fixed_vars)
        self.traj_initial = {key: val[0] for key, val in self.traj_group.ST.items()
                             if not key.startswith('_')}
        self.traj_net = core.Network(self.traj_group)

    def plot_trajectory(self, initials, duration, plot_duration=None, inputs=(), show=False):
        """Plot trajectories according to the settings.

        Parameters
        ----------
        initials : list, tuple
            The initial value setting of the targets. It can be a tuple/list of floats to specify
            each value of dynamical variables (for example, ``(a, b)``). It can also be a
            tuple/list of tuple to specify multiple initial values (for example,
            ``[(a1, b1), (a2, b2)]``).
        duration : int, float, tuple, list
            The running duration. Same with the ``duration`` in ``NeuGroup.run()``.
            It can be a int/float (``t_end``) to specify the same running end time,
            or it can be a tuple/list of int/float (``(t_start, t_end)``) to specify
            the start and end simulation time. Or, it can be a list of tuple
            (``[(t1_start, t1_end), (t2_start, t2_end)]``) to specify the specific
            start and end simulation time for each initial value.
        plot_duration : tuple/list of tuple, optional
            The duration to plot. It can be a tuple with ``(start, end)``. It can
            also be a list of tuple ``[(start1, end1), (start2, end2)]`` to specify
            the plot duration for each initial value running.
        inputs : tuple, list
            The inputs to the model. Same with the ``inputs`` in ``NeuGroup.run()``
        show : bool
            Whether show or not.
        """

        # 1. format the initial values
        all_vars = self.fast_var_names + self.slow_var_names
        if isinstance(initials, dict):
            initials = [initials]
        elif isinstance(initials, (list, tuple)):
            if isinstance(initials[0], (int, float)):
                initials = [{all_vars[i]: v for i, v in enumerate(initials)}]
            elif isinstance(initials[0], dict):
                initials = initials
            elif isinstance(initials[0], (tuple, list)) and isinstance(initials[0][0], (int, float)):
                initials = [{all_vars[i]: v for i, v in enumerate(init)} for init in initials]
            else:
                raise ValueError
        else:
            raise ValueError
        for initial in initials:
            if len(initial) != len(all_vars):
                raise errors.AnalyzerError(f'Should provide all fast-slow variables ({all_vars}) '
                                           f' initial values, but we only get initial values for '
                                           f'variables {list(initial.keys())}.')

        # 2. format the running duration
        if isinstance(duration, (int, float)):
            duration = [(0, duration) for _ in range(len(initials))]
        elif isinstance(duration[0], (int, float)):
            duration = [duration for _ in range(len(initials))]
        else:
            assert len(duration) == len(initials)

        # 3. format the plot duration
        if plot_duration is None:
            plot_duration = duration
        if isinstance(plot_duration[0], (int, float)):
            plot_duration = [plot_duration for _ in range(len(initials))]
        else:
            assert len(plot_duration) == len(initials)

        # 4. format the inputs
        if len(inputs):
            if isinstance(inputs[0], (tuple, list)):
                inputs = [(self.traj_group,) + tuple(input) for input in inputs]
            elif isinstance(inputs[0], str):
                inputs = [(self.traj_group,) + tuple(inputs)]
            else:
                raise errors.ModelUseError()

        # 5. run the network
        for init_i, initial in enumerate(initials):
            #   5.1 set the initial value
            for key, val in self.traj_initial.items():
                self.traj_group.ST[key] = val
            for key in all_vars:
                self.traj_group.ST[key] = initial[key]
            for key, val in self.fixed_vars.items():
                if key in self.traj_group.ST:
                    self.traj_group.ST[key] = val

            #   5.2 run the model
            self.traj_net.run(duration=duration[init_i], inputs=inputs,
                              report=False, data_to_host=True, verbose=False)

            #   5.3 legend
            legend = 'traj, '
            for key in all_vars:
                legend += f'${key}_{init_i}$={initial[key]}, '
            legend = legend[:-2]

            #   5.4 trajectory
            start = int(plot_duration[init_i][0] / profile.get_dt())
            end = int(plot_duration[init_i][1] / profile.get_dt())

            #   5.5 visualization
            for var_name in self.fast_var_names:
                s0 = self.traj_group.mon[self.slow_var_names[0]][start: end, 0]
                fast = self.traj_group.mon[var_name][start: end, 0]

                fig = plt.figure(var_name)
                if len(self.slow_var_names) == 1:
                    plt.plot(s0, fast, label=legend)
                    # middle = int(s0.shape[0] / 2)
                    # plt.arrow(s0[middle], fast[middle],
                    #           s0[middle + 1] - s0[middle], fast[middle + 1] - fast[middle],
                    #           shape='full')

                elif len(self.slow_var_names) == 2:
                    fig.gca(projection='3d')
                    s1 = self.traj_group.mon[self.slow_var_names[1]][start: end, 0]
                    plt.plot(s0, s1, fast, label=legend)
                else:
                    raise errors.AnalyzerError

        # 6. visualization
        for var_name in self.fast_vars.keys():
            fig = plt.figure(var_name)

            # scale = (self.lim_scale - 1.) / 2
            if len(self.slow_var_names) == 1:
                # plt.xlim(*utils.rescale(self.slow_vars[self.slow_var_names[0]], scale=scale))
                # plt.ylim(*utils.rescale(self.fast_vars[var_name], scale=scale))
                plt.xlabel(self.slow_var_names[0])
                plt.ylabel(var_name)
            elif len(self.slow_var_names) == 2:
                ax = fig.gca(projection='3d')
                # ax.set_xlim(*utils.rescale(self.slow_vars[self.slow_var_names[0]], scale=scale))
                # ax.set_ylim(*utils.rescale(self.slow_vars[self.slow_var_names[1]], scale=scale))
                # ax.set_zlim(*utils.rescale(self.fast_vars[var_name], scale=scale))
                ax.set_xlabel(self.slow_var_names[0])
                ax.set_ylabel(self.slow_var_names[1])
                ax.set_zlabel(var_name)

            plt.legend()

        if show:
            plt.show()


class _FastSlow1D(_Bifurcation1D):
    def __init__(self, model, fast_vars, slow_vars, fixed_vars=None,
                 pars_update=None, numerical_resolution=0.1, options=None):
        super(_FastSlow1D, self).__init__(model=model,
                                          target_pars=slow_vars,
                                          target_vars=fast_vars,
                                          fixed_vars=fixed_vars,
                                          pars_update=pars_update,
                                          numerical_resolution=numerical_resolution,
                                          options=options)
        self.traj = _FastSlowTrajectory(model=model,
                                        fast_vars=fast_vars,
                                        slow_vars=slow_vars,
                                        fixed_vars=fixed_vars,
                                        pars_update=pars_update,
                                        numerical_resolution=numerical_resolution,
                                        options=options)

    def plot_trajectory(self, *args, **kwargs):
        self.traj.plot_trajectory(*args, **kwargs)


class _FastSlow2D(_Bifurcation2D, ):
    def __init__(self, model, fast_vars, slow_vars, fixed_vars=None,
                 pars_update=None, numerical_resolution=0.1, options=None):
        super(_FastSlow2D, self).__init__(model=model,
                                          target_pars=slow_vars,
                                          target_vars=fast_vars,
                                          fixed_vars=fixed_vars,
                                          pars_update=pars_update,
                                          numerical_resolution=numerical_resolution,
                                          options=options)
        self.traj = _FastSlowTrajectory(model=model,
                                        fast_vars=fast_vars,
                                        slow_vars=slow_vars,
                                        fixed_vars=fixed_vars,
                                        pars_update=pars_update,
                                        numerical_resolution=numerical_resolution,
                                        options=options)

    def plot_trajectory(self, *args, **kwargs):
        self.traj.plot_trajectory(*args, **kwargs)


if __name__ == '__main__':
    Axes3D
