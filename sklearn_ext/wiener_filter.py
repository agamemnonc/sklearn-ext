import numpy as np


from scipy.linalg import toeplitz
from scipy.signal import lfilter

from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import check_X_y
from sklearn.metrics import r2_score


class WienerFilter(object):
    """Wiener Filter regression.

    Parameters
    ----------

    reg_lambda : float
        Regularization constant
    n_lags : int
        Maximum length of filters (i.e. number of lags) used for regression.

    Attributes
    ----------

    n_features : int
        Number of features
    n_outputs : int
        Number of outputs
    coef_ : array-like, shape (n_features*n_lags, n_outputs)
        Coefficient matrix
    intercept_ : array, shape (n_outputs)
        Independent term in the linear model.
    """

    def __init__(self, n_features, n_outputs, reg_lambda=1e-4, n_lags=1):
        self.reg_lambda = reg_lambda
        self.n_lags = n_lags

    def _covf(self, x, M):
        n_sam, n_dim = np.shape(x)
        x = np.vstack((x, np.zeros((M, n_dim))))
        rows = np.arange(n_sam)
        R = np.zeros((n_dim**2, M), dtype=float)
        for jj in range(M):
            a = np.dot(np.transpose(x[rows, :]), x[rows + jj, :])
            R[:, jj] = (np.conj(a) / n_sam).reshape((n_dim**2), order='F')
        return R

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False,
                         y_numeric=True, multi_output=True)
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        output_mean = np.mean(y, axis=0)
        y = np.subtract(y, output_mean)
        numio = self.n_features + self.n_outputs_
        R = self._covf(np.hstack((X, y)), self.n_lags)
        PHI = np.empty((2 * self.n_lags - 1, numio**2), dtype=float, order='C')
        for ii in range(numio):
            for jj in range(numio):
                PHI[:, ii +
                    jj *
                    numio] = np.hstack((R[jj +
                                          ii *
                                          numio, np.arange(self.n_lags -
                                                           1, 0, -
                                                           1)], R[ii +
                                                                  jj *
                                                                  numio, :]))

        Nxxr = np.arange(self.n_lags - 1, 2 * (self.n_lags - 1) + 1, 1)
        Nxxc = np.arange(self.n_lags - 1, -1, -1)
        Nxy = np.arange(self.n_lags - 1, 2 * (self.n_lags - 1) + 1)
        # Solve matrix equations to identify filters
        PX = np.empty(
            (self.n_features_ *
             self.n_lags,
             self.n_features_ *
             self.n_lags),
            dtype=float,
            order='C')
        for ii in range(self.n_features_):
            for jj in range(self.n_features_):
                c_start = ii * self.n_lags
                c_end = (ii + 1) * self.n_lags
                r_start = jj * self.n_lags
                r_end = (jj + 1) * self.n_lags
                PX[r_start:r_end, c_start:c_end] = toeplitz(
                    PHI[Nxxc, ii + (jj) * numio], PHI[Nxxr, ii + (jj) * numio])

        PXY = np.empty(
            (self.n_features_ *
             self.n_lags,
             self.n_outputs_),
            dtype=float,
            order='C')
        for ii in range(self.n_features_):
            for jj in range(self.n_features_,
                            self.n_features_ + self.n_outputs_, 1):
                r_start = ii * self.n_lags
                r_end = (ii + 1) * self.n_lags
                c_ind = jj - self.n_features_
                PXY[r_start:r_end, c_ind] = PHI[Nxy, ii + (jj) * numio]

        self.coef_ = np.linalg.solve(
            (PX + self.reg_lambda * np.identity(PX.shape[0])), PXY)
        self.intercept_ = output_mean

    def predict(self, X, batch=False):
        """
        If batch is True, X.shape = (num_sam, n_features), if not
        X.shape = (n_lags, n_features).
        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, "coef_")
        if batch is True:
            X_ud = np.flipud(X)
            y = np.dot(X_ud.reshape(-1, order='F'), self.coef_)
        else:
            n_samples = X.shape[0]
            y = np.zeros((n_samples, self.n_outputs_))
            for ii in range(self.n_outputs_):
                for jj in range(self.n_features_):
                    coef = self.coef_[
                        jj *
                        self.n_lags:(
                            jj +
                            1) *
                        self.n_lags,
                        ii]
                    y[:, ii] += lfilter(coef, 1, X[:, jj], axis=-1)

            y = y[self.n_lags - 1:, :]
        return y + self.intercept_

    def score(self, X, y, batch=False, multioutput='uniform_average'):
        self.vaf = r2_score(y[self.n_lags - 1:, :],
                            self.predict(X), multioutput)
