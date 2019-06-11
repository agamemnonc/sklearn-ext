import numpy as np


from scipy.linalg import toeplitz
from scipy.signal import lfilter

from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import check_X_y
from sklearn.metrics import r2_score


__all__ = ['WienerFilter']


def _covf(X, M):
    """Estimate time-series covariance functions.

    Parameters
    ----------
    X: array-like, shape = (n_samples, n_features)
        Data from which to compute the covariance estimate
    M : int
        The maximum delay - 1, for which the covariance function is
        estimated

    Returns
    -------
    covariance : array, shape = (n_features ** 2, M)
        Covariance matrix
    """

    n_samples, n_features = np.shape(X)
    X = np.vstack((X, np.zeros((M, n_features))))
    rows = np.arange(n_samples)
    covariance = np.zeros((n_features**2, M), dtype=float)
    for jj in range(M):
        a = np.dot(np.transpose(X[rows, :]), X[rows + jj, :])
        covariance[:, jj] = (np.conj(a) / n_samples).reshape(
            (n_features**2), order='F')
    return covariance


class WienerFilter(BaseEstimator, RegressorMixin):
    """Wiener Filter regression.

    Parameters
    ----------
    reg_lambda : float
        Regularization constant
    n_lags : int
        Maximum length of filters (i.e. number of lags) used for regression.

    Attributes
    ----------
    coef_ : array-like, shape (n_features*n_lags, n_outputs)
        Coefficient matrix
    intercept_ : array, shape (n_outputs)
        Independent term in the linear model.
    """

    def __init__(self, reg_lambda=1e-4, n_lags=1):
        self.reg_lambda = reg_lambda
        self.n_lags = n_lags

    def fit(self, X, y):
        """
        Fit linear model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : array-like, shape = (n_samples, n_outputs)
            Target values

        Returns
        -------
        self : returns an instance of self.
        """

        X, y = check_X_y(X, y, accept_sparse=False,
                         y_numeric=True, multi_output=True)
        n_features_ = X.shape[1]
        n_outputs_ = y.shape[1]
        output_mean = np.mean(y, axis=0)
        y = np.subtract(y, output_mean)
        numio = n_features_ + n_outputs_
        R = _covf(np.hstack((X, y)), self.n_lags)
        PHI = np.empty((2 * self.n_lags - 1, numio**2), dtype=float, order='C')
        for ii in range(numio):
            for jj in range(numio):
                PHI[:, ii +
                    jj *
                    numio] = np.hstack((
                        R[jj + ii * numio,
                          np.arange(self.n_lags - 1, 0, -1)],
                        R[ii + jj * numio, :]))

        Nxxr = np.arange(self.n_lags - 1, 2 * (self.n_lags - 1) + 1, 1)
        Nxxc = np.arange(self.n_lags - 1, -1, -1)
        Nxy = np.arange(self.n_lags - 1, 2 * (self.n_lags - 1) + 1)
        # Solve matrix equations to identify filters
        PX = np.empty(
            (n_features_ *
             self.n_lags,
             n_features_ *
             self.n_lags),
            dtype=float,
            order='C')
        for ii in range(n_features_):
            for jj in range(n_features_):
                c_start = ii * self.n_lags
                c_end = (ii + 1) * self.n_lags
                r_start = jj * self.n_lags
                r_end = (jj + 1) * self.n_lags
                PX[r_start:r_end, c_start:c_end] = toeplitz(
                    PHI[Nxxc, ii + (jj) * numio], PHI[Nxxr, ii + (jj) * numio])

        PXY = np.empty(
            (n_features_ *
             self.n_lags,
             n_outputs_),
            dtype=float,
            order='C')
        for ii in range(n_features_):
            for jj in range(n_features_,
                            n_features_ + n_outputs_, 1):
                r_start = ii * self.n_lags
                r_end = (ii + 1) * self.n_lags
                c_ind = jj - n_features_
                PXY[r_start:r_end, c_ind] = PHI[Nxy, ii + (jj) * numio]

        self.coef_ = np.linalg.solve(
            (PX + self.reg_lambda * np.identity(PX.shape[0])), PXY)
        self.intercept_ = output_mean

        return self

    def predict(self, X, batch=True):
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features) or (n_lags, n_features)
            The input samples.
        batch : boolean, optional, default True
            If True, a batch prediction is made. Otherwise, a single prediction
            is made. In the latter case, data in X should be in augmented form,
            i.e., the shape of X should be (n_lags, n_features), where the most
            recent observations are stored in the last row of the array.

        Returns
        -------
        y : array, shape = (n_samples,n_outputs)
            The predicted values.
        """

        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "coef_")
        n_features = X.shape[1]
        n_outputs = self.intercept_.size
        if batch is False:
            X_ud = np.flipud(X)
            y = np.dot(X_ud.reshape(-1, order='F'), self.coef_)
        else:
            n_samples = X.shape[0]
            y = np.zeros((n_samples, n_outputs))
            for ii in range(n_outputs):
                for jj in range(n_features):
                    coef = self.coef_[
                        jj * self.n_lags:(jj + 1) * self.n_lags, ii]
                    y[:, ii] += lfilter(coef, 1, X[:, jj], axis=-1)
            y = y[self.n_lags - 1:, :]

        return y + self.intercept_

    def score(self, X, y, sample_weight=None, multioutput='uniform_average'):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.

        Notes
        -----
        This method can only be used for batch prediction, since R^2 does not
        make sense for a single prediction.
        """

        return r2_score(
            y[self.n_lags - 1:, :],
            self.predict(X, batch=True),
            sample_weight=sample_weight,
            multioutput=multioutput)
