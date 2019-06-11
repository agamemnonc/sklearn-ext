"""
Regularized Discriminant Analysis
"""

import warnings

import numpy as np
from sklearn.externals.six import string_types
from sklearn.externals.six.moves import xrange

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import ledoit_wolf, empirical_covariance
from sklearn.covariance import shrunk_covariance
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import StandardScaler


__all__ = ['RegularizedDiscriminantAnalysis']


def _cov(X, shrinkage=None):
    """Estimate covariance matrix (using optional shrinkage).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    shrinkage : string or float, optional
        Shrinkage parameter, possible values:
          - None or 'empirical': no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

    Returns
    -------
    s : array, shape (n_features, n_features)
        Estimated covariance matrix.
    """
    shrinkage = "empirical" if shrinkage is None else shrinkage
    if isinstance(shrinkage, string_types):
        if shrinkage == 'auto':
            sc = StandardScaler()  # standardize features
            X = sc.fit_transform(X)
            s = ledoit_wolf(X)[0]
            s = sc.scale_[:, np.newaxis] * s * \
                sc.scale_[np.newaxis, :]  # rescale
        elif shrinkage == 'empirical':
            s = empirical_covariance(X)
        else:
            raise ValueError('unknown shrinkage parameter')
    elif isinstance(shrinkage, float) or isinstance(shrinkage, int):
        if shrinkage < 0 or shrinkage > 1:
            raise ValueError('shrinkage parameter must be between 0 and 1')
        s = shrunk_covariance(empirical_covariance(X), shrinkage)
    else:
        raise TypeError('shrinkage must be of string or int type')
    return s


def _class_means(X, y):
    """Compute class means.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    means : array-like, shape (n_features,)
        Class means.
    """
    means = []
    classes = np.unique(y)
    for group in classes:
        Xg = X[y == group, :]
        means.append(Xg.mean(0))
    return np.asarray(means)


def _class_cov(X, y, priors=None, shrinkage=None):
    """Compute class covariance matrix.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.

    priors : array-like, shape (n_classes,)
        Class priors.

    shrinkage : string or float, optional
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

    Returns
    -------
    cov : array-like, shape (n_features, n_features)
        Class covariance matrix.
    """
    classes = np.unique(y)
    covs = []
    for group in classes:
        Xg = X[y == group, :]
        covs.append(np.atleast_2d(_cov(Xg, shrinkage)))
    return np.average(covs, axis=0, weights=priors)


class RegularizedDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    """
    Regularized Discriminant Analysis

    A classifier with a quadratic decision boundary, generated
    by fitting class conditional densities to the data
    and using Bayes' rule.

    The model fits a Gaussian density to each class. The covariance matrix
    for each class is a compromise between the sample estimate for the
    particular class and the pooled covariance matrix.


    Parameters
    ----------
    priors : array, optional, shape = [n_classes]
        Priors on classes

    reg_param_alpha : float, optional
        Regularizes the covariance estimate of each class Sigma_k as
        ``reg_param_alpha*Sigma_k + (1-a)*Sigma

    reg_param_gamma : float, optional
        Regularizes the covariance estimate as
        ``(1-reg_param_gamma)*Sigma + reg_param_gamma*np.eye(n_features)``.
        Applies to both the pooled and class-specific covariance matrices.

    shrinkage : string or float, optional
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.


    Attributes
    ----------
    covariances_ : list of array-like, shape = [n_features, n_features]
        Covariance matrices of each class.

    pooled_covariance_ : pooled covariance matrix

    means_ : array-like, shape = [n_classes, n_features]
        Class means.

    priors_ : array-like, shape = [n_classes]
        Class priors (sum to 1).

    quad_coef_ : array, shape (n_features, n_features)
        Quadratic coefficients

    linear_coef_ : array, shape (n_features,)
        Linear coefficients .

    intercept_ : array, shape (n_features,)
        Intercept term.



    tol : float, optional, default 1.0e-4
        Threshold used for rank estimation.


    Examples
    --------
    >>> from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = RegularizedDiscriminantAnalysis()
    >>> clf.fit(X, y)
    ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    RegularizedDiscriminantAnalysis(priors=None, reg_param_alpha=0.5,
                                    reg_param_gamma=0.0, shrinkage=None,
                                    tol=0.0001)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]

    See also
    --------
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis: Linear
        Discriminant Analysis
    sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis: Quadratic
        Discriminant Analysis
    """

    def __init__(self, priors=None, reg_param_alpha=0., reg_param_gamma=0.,
                 shrinkage=None, tol=1.0e-4):
        self.priors = np.asarray(priors) if priors is not None else None
        self.reg_param_alpha = reg_param_alpha
        self.reg_param_gamma = reg_param_gamma
        self.shrinkage = shrinkage
        self.tol = tol

    def fit(self, X, y, tol=None):
        """Fit the model according to the given training data and parameters.


        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array, shape = [n_samples]
            Target values (integers)
        """
        if tol:
            warnings.warn("The parameter 'tol' is deprecated as of version "
                          "0.17 and will be removed in 0.19. The parameter is "
                          "no longer necessary because the value is set via "
                          "the estimator initialisation or set_params method.",
                          DeprecationWarning)
            self.tol = tol
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        pooledcov = _class_cov(X, y, self.priors_, self.shrinkage)
        self.pooled_covariance_ = (1 - self.reg_param_gamma) * \
            pooledcov + self.reg_param_gamma * np.diag(np.diag(pooledcov))
        cov = []
        means = []
        quad_coef = []
        linear_coef = []
        intercept = []
        for ind in xrange(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            if len(Xg) == 1:
                raise ValueError('y has only 1 sample in class %s, covariance '
                                 'is ill defined.' % str(self.classes_[ind]))

            covg = np.atleast_2d(_cov(Xg, self.shrinkage))
            covg = (1 - self.reg_param_gamma) * covg + \
                self.reg_param_gamma * np.diag(np.diag(covg))
            covg = self.reg_param_alpha * covg + \
                (1 - self.reg_param_alpha) * self.pooled_covariance_

            U, S, V = np.linalg.svd(covg)
            temp1 = np.dot(U, np.dot(np.diag(1 / S), U.T))
            temp2 = np.dot(meang, temp1)

            intercept.append(-0.5 * (np.dot(temp2, meang) +
                                     np.linalg.slogdet(covg)[1]) +
                             np.log(self.priors_[ind]))
            linear_coef.append(temp2)
            quad_coef.append(-0.5 * temp1)
            means.append(meang)
            cov.append(covg)

        self.means_ = np.asarray(means)
        self.covariances_ = np.asarray(cov)
        self.quad_coef_ = np.asarray(quad_coef)
        self.linear_coef_ = np.asarray(linear_coef)
        self.intercept_ = np.asarray(intercept)
        return self

    def _decision_function(self, X):
        check_is_fitted(self, 'classes_')

        X = check_array(X)
        norm2 = []
        for i in range(len(self.classes_)):
            norm2.append(self.intercept_[i] +
                         np.dot(self.linear_coef_[i], X.T) +
                         np.diag(np.dot(X, np.dot(self.quad_coef_[i], X.T))))
        norm2 = np.array(norm2).T   # shape = [len(X), n_classes]
        return norm2

    def decision_function(self, X):
        """Apply decision function to an array of samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Array of samples (test vectors).

        Returns
        -------
        C : array, shape = [n_samples, n_classes] or [n_samples,]
            Decision function values related to each class, per sample.
            In the two-class case, the shape is [n_samples,], giving the
            log likelihood ratio of the positive class.
        """
        dec_func = self._decision_function(X)
        # handle special case of two classes
        if len(self.classes_) == 2:
            return dec_func[:, 1] - dec_func[:, 0]
        return dec_func

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        The predicted class C for each sample in X is returned.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
        """
        d = self._decision_function(X)
        y_pred = self.classes_.take(d.argmax(1))
        return y_pred

    def predict_proba(self, X):
        """Return posterior probabilities of classification.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Array of samples/test vectors.

        Returns
        -------
        C : array, shape = [n_samples, n_classes]
            Posterior probabilities of classification per class.
        """
        values = self._decision_function(X)
        # compute the likelihood of the underlying gaussian models
        # up to a multiplicative constant.
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        # compute posterior probabilities
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

    def predict_log_proba(self, X):
        """Return posterior probabilities of classification.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Array of samples/test vectors.

        Returns
        -------
        C : array, shape = [n_samples, n_classes]
            Posterior log-probabilities of classification per class.
        """
        probas_ = self.predict_proba(X)
        return np.log(probas_)
