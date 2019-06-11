"""
Smoothing algorithms.
"""
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from abc import abstractmethod


__all__ = [
    'MovingAverageSmoothing',
    'SingleExponentialSmoothing',
    'DoubleExponentialSmoothing'
    ]

class _BaseSmoothing(BaseEstimator, TransformerMixin):
    """Base class for smoothers.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    @abstractmethod
    def fit(self):
        """Smoother classes do not need to implement a ``fit`` method.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    @abstractmethod
    def _smoothing_step(self, x):
        """Smoothing function. Subclasses should implement this method.
        """

    def transform(self, X):
        """Smooths data matrix one sample at a time.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to transform.
        """
        X = check_array(X)
        X_smoothed = np.zeros_like(X)
        for i, sample in enumerate(X):
            X_smoothed[i, :] = self._smoothing_step(sample)

        return X_smoothed

    def fit_transform(self, x):
        return self.transform(x)

    @abstractmethod
    def reset(self):
        """Resets the smoother. Subclasses should implement this method.
        """


class MovingAverageSmoothing(_BaseSmoothing):
    """Moving average smoothing.

    Parameters
    ----------
    k : int
        Number of previous samples to use for smoothing.

    weights : array, optional (default ones), shape = (k,)
        Weight vector. First element corresponds to weight for most recent
        observation.
    """

    def __init__(self, k, weights=None):

        self.k = self._check_k(k)
        self.weights = self._check_weights(weights)
        self.cache_ = []

    def _smoothing_step(self, x):
        """Smoothing function.

        Parameters
        ----------
        X : array-like, shape (n_features,)
            Sample to smooth.

        Returns
        -------
        x : array-like, shape (n_features,)
            Smoothed sample.
        """
        self.cache_.append(x)
        if len(self.cache_) > self.k:
            self.cache_ = self.cache_[1:]

        return np.dot(np.flipud(self.weights[:len(self.cache_)]),
                      np.asarray(self.cache_)) / len(self.cache_)

    def _check_k(self, k):
        """Checks number of samples k.

        Parameters
        ----------
        k : int
            Number of samples.

        Returns
        -------
        k : int
            Number of samples.

        Raises
        ------
        ValueError
            If k < 1.
        """
        if k < 1:
            raise ValueError("Number of samples must be > 0, but {} was " \
                             "provided.".format(k))

        return k

    def _check_weights(self, weights):
        """Checks weight vector.

        Parameters
        ----------
        weights : array-like, shape (k,)
            Weight vector.

        Raises
        ------
        ValueError
            If the weight vector does not agree with the number of samples used
            for smoothing.
        """
        if weights is None:
            weights = np.ones((self.k,))
        else:
            weights = check_array(weights, ensure_2d=False)
            if weights.size != self.k:
                raise ValueError("Weight array must have shape ({},)".format(
                    self.k))

        return weights

    def reset(self):
        """Resets the smoother. """
        self.cache_ = []

class _ExponentialSmoothing(_BaseSmoothing):
    """Base class for exponential smoothing functions.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    def _check_smoothing_parameter(self, parameter):
        """Ensures smoothing parameter is within the range [0,1].

        Parameters
        ----------
        parameter : float
            Smoothing parameter.

        Returns
        -------
        parameter : float
            The parameter itelf.

        Raises
        ------
        ValueError
            If the parameter is not whithing the range [0,1].
        """
        if (parameter < 0. or parameter > 1.):
            raise ValueError("Smoothing parameter must be between 0 and 1.")

        return parameter


class SingleExponentialSmoothing(_ExponentialSmoothing):
    """Single exponential smoothing.

    s_0 = x_0

    For t > 0:
    s_t = \alpha * x_t + \left(1 - \alpha\right) * s_{t-1}
    """
    def __init__(self, alpha):
        self.alpha = self._check_smoothing_parameter(alpha)
        self.cache_ = None

    def _smoothing_step(self, x):
        if self.cache_ is None: # t=0
            out = x.copy()
            self.cache_ = out
        else:
            out = self.alpha*x + (1-self.alpha) * self.cache_
            self.cache_ = out

        return out

    def reset(self):
        """Resets the smoother. """
        self.cache_ = None


class DoubleExponentialSmoothing(_ExponentialSmoothing):
    """Double exponential smoothing.

    s_1 = x_1
    b_1 = x_1 - x_0

    For t > 1:
    s_t = \alpha * x_t + \left(1 - \alpha \right) *
        \left(s_{t-1} + b_{t-1} \right)
    b_t = \beta \left(s_t - s_{t-1}\right) +
        \left(1 - \beta\right) * b_{t-1}
    """
    def __init__(self, alpha, beta):
        self.alpha = self._check_smoothing_parameter(alpha)
        self.beta = self._check_smoothing_parameter(beta)
        self.cache_ = [None] * 2

    def _smoothing_step(self, x):
        if self.cache_[1] is None: # t < 2
            if self.cache_[0] is None: # t = 0
                out = x
                self.cache_[0] = out
            else: # t = 1
                out = x
                self.cache_[1] = out - self.cache_[0]
                self.cache_[0] = out
        else:
            out = self.alpha*x + (1-self.alpha) * \
                (self.cache_[0]+self.cache_[1])

            self.cache_[1] = self.beta * (out-self.cache_[0]) + \
                (1-self.beta) * self.cache_[1]
            self.cache_[0] = out

        return out

    def reset(self):
        """Resets the smoother. """
        self.cache_ = [None] * 2
