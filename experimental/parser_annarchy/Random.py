import numpy as np
import logging

distributions_arguments = {
    'Uniform' : 2,
    'DiscreteUniform': 2,
    'Normal' : 2,
    'LogNormal': 2,
    'Exponential': 1,
    'Gamma': 2
}

distributions_equivalents = {
    'Uniform' : 'std::uniform_real_distribution< %(float_prec)s >',
    'DiscreteUniform': 'std::uniform_int_distribution<int>',
    'Normal' : 'std::normal_distribution< %(float_prec)s >',
    'LogNormal': 'std::lognormal_distribution< %(float_prec)s >',
    'Exponential': 'std::exponential_distribution< %(float_prec)s >',
    'Gamma': 'std::gamma_distribution< %(float_prec)s >'
}

# List of available distributions
available_distributions = distributions_arguments.keys()

class RandomDistribution(object):
    """
    BaseClass for random distributions.
    """

    def get_values(self, shape):
        """
        Returns a bnp.ndarray with the given shape
        """
        logging.error('instantiated base class RandomDistribution is not allowed.')
        return 0.0

    def get_list_values(self, size):
        """
        Returns a list of the given size.
        """
        return list(self.get_values(size))

    def get_value(self):
        """
        Returns a single float value.
        """
        return self.get_values((1))[0]

    def keywords(self):
        return available_distributions

    def latex(self):
        return '?'

class Uniform(RandomDistribution):
    """
    Random distribution object using the uniform distribution between ``min`` and ``max``.

    The returned values are floats in the range [min, max].
    """
    def __init__(self, min, max):
        """
        *Parameters*:

        * **min**: minimum value.

        * **max**: maximum value.
        """
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a Numpy array with the given shape.
        """
        return np.random.uniform(self.min, self.max, shape)

    def latex(self):
        return "$\\mathcal{U}$(" + str(self.min) + ', ' + str(self.max) + ')'


class DiscreteUniform(RandomDistribution):
    """
    Random distribution object using the discrete uniform distribution between ``min`` and ``max``.

    The returned values are integers in the range [min, max].
    """
    def __init__(self, min, max):
        """
        *Parameters*:

        * **min**: minimum value

        * **max**: maximum value
        """
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a bnp.ndarray with the given shape
        """
        return np.random.random_integers(self.min, self.max, shape)

    def latex(self):
        return "$\\mathcal{U}$(" + str(self.min) + ', ' + str(self.max) + ')'


class Normal(RandomDistribution):
    """
    Random distribution instance returning a random value based on a normal (Gaussian) distribution.
    """
    def __init__(self, mu, sigma, min=None, max=None):
        """
        *Parameters*:

        * **mu**: mean of the distribution

        * **sigma**: standard deviation of the distribution

        * **seed**: seed for the random number generator. By default, the seed takes the value defined in ``ANNarchy.setup()``.

        * **min**: minimum value returned (default: unlimited).

        * **max**: maximum value returned (default: unlimited).
        """
        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a bnp.ndarray with the given shape
        """
        data = np.random.normal(self.mu, self.sigma, shape)
        if self.min != None:
            data[data<self.min] = self.min
        if self.max != None:
            data[data>self.max] = self.max
        return data

    def latex(self):
        return "$\\mathcal{N}$(" + str(self.mu) + ', ' + str(self.sigma) + ')'

class LogNormal(RandomDistribution):
    """
    Random distribution instance returning a random value based on lognormal distribution.
    """
    def __init__(self, mu, sigma, min=None, max=None):
        """
        *Parameters*:

        * **mu**: mean of the distribution

        * **sigma**: standard deviation of the distribution

        * **min**: minimum value returned (default: unlimited).

        * **max**: maximum value returned (default: unlimited).
        """
        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a bnp.ndarray with the given shape
        """
        data = np.random.lognormal(self.mu, self.sigma, shape)
        if self.min != None:
            data[data<self.min] = self.min
        if self.max != None:
            data[data>self.max] = self.max
        return data

    def latex(self):
        return "$\\ln\\mathcal{N}$(" + str(self.mu) + ', ' + str(self.sigma) + ')'

class Exponential(RandomDistribution):
    """
    Random distribution instance returning a random value based on exponential distribution, according the density function:

    .. math ::

        P(x | \lambda) = \lambda e^{(-\lambda x )}

    """
    def __init__(self, Lambda, min=None, max=None):
        """
        *Parameters*:

        * **Lambda**: rate parameter.

        * **seed**: seed for the random number generator. By default, the seed takes the value defined in ``ANNarchy.setup()``.

        * **min**: minimum value returned (default: unlimited).

        * **max**: maximum value returned (default: unlimited).

        .. note::

            ``Lambda`` is capitalized, otherwise it would be a reserved Python keyword.

        """
        self.Lambda = Lambda
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a bnp.ndarray with the given shape.
        """
        data = np.random.exponential(self.Lambda, shape)
        if self.min != None:
            data[data<self.min] = self.min
        if self.max != None:
            data[data>self.max] = self.max
        return data

    def latex(self):
        return "$\\exp$(" + str(self.Lambda) + ')'

class Gamma(RandomDistribution):
    """
    Random distribution instance returning a random value based on gamma distribution.
    """
    def __init__(self, alpha, beta=1.0, seed=-1, min=None, max=None):
        """
        *Parameters*:

        * **alpha**: shape of the gamma distribution

        * **beta**: scale of the gamma distribution

        * **min**: minimum value returned (default: unlimited).

        * **max**: maximum value returned (default: unlimited).
        """
        self.alpha = alpha
        self.beta = beta
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a bnp.ndarray with the given shape
        """
        data = np.random.gamma(self.alpha, self.beta, shape)
        if self.min != None:
            data[data<self.min] = self.min
        if self.max != None:
            data[data>self.max] = self.max
        return data

    def latex(self):
        return "$\\Gamma$(" + str(self.alpha) + ', ' + str(self.beta) + ')'
