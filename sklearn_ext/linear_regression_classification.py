import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets


__all__ = ['LinearRegressionClassifier']


class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):
    """Linear Regression Classifier

    The classifier works by comparing the distance of a test point to the
    subspace spanned by the collection of training points in each class. It
    computes the distance for each class and picks the class with minimum
    distance. Prediction probabilities are not natively supported.

    Attributes
    ----------
    hat_ : list of array-like, shape = [n_features, n_features]
        Hat matrix for each class.

    References
    ----------
    .. [1] N. Imran, R. Togneri, M. Bennamoun.
           "Linear regression for face recognition." IEEE transactions on
           pattern analysis and machine intelligence. 32(11), 2106-2112, 2010.
    """

    def __init__(self):
        super(LinearRegressionClassifier, self).__init__()

    def fit(self, X, y):
        """Fit the model according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array, shape = [n_samples]
            Target values.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        _, counts = np.unique(y, return_counts=True)
        if np.any(counts > n_features):
            warnings.warn("Found some classes with more counts than input "
                          "features. Results may be unstable.")

        self.hat_ = []

        for ind in range(n_classes):
            Xg = X[y == ind, :]
            Gg = np.dot(Xg, Xg.T)
            self.hat_.append(np.dot(np.dot(Xg.T, np.linalg.inv(Gg)), Xg))

        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        The predicted class C for each sample in X is returned.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples,]
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        D = np.zeros((n_samples, n_classes))
        for ind in range(n_classes):
            D[:, ind] = np.linalg.norm(
                np.dot(X, np.eye(n_features) - self.hat_[ind]),
                axis=1)

        return np.argmin(D, axis=1)
