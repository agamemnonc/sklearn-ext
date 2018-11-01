"""
ClassifierChain predict_proba and decision_function fixes
"""

import numpy as np
import scipy.sparse as sp
from sklearn.base import MetaEstimatorMixin
from sklearn.base import ClassifierMixin
from sklearn.utils import check_array
from sklearn.multioutput import _BaseChain

from sklearn.utils.metaestimators import if_delegate_has_method

class ClassifierChain(_BaseChain, ClassifierMixin, MetaEstimatorMixin):
    """A multi-label model that arranges binary classifiers into a chain.
    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.
    Read more in the :ref:`User Guide <classifierchain>`.
    Parameters
    ----------
    base_estimator : estimator
        The base estimator from which the classifier chain is built.
    order : array-like, shape=[n_outputs] or 'random', optional
        By default the order will be determined by the order of columns in
        the label matrix Y.::
            order = [0, 1, 2, ..., Y.shape[1] - 1]
        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::
            order = [1, 3, 2, 4, 0]
        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.
        If order is 'random' a random ordering will be used.
    cv : int, cross-validation generator or an iterable, optional \
    (default=None)
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        If cv is None the true labels are used when fitting. Otherwise
        possible inputs for cv are:
        * integer, to specify the number of folds in a (Stratified)KFold,
        * An object to be used as a cross-validation generator.
        * An iterable yielding train, test splits.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
        The random number generator is used to generate random chain orders.
    Attributes
    ----------
    classes_ : list
        A list of arrays of length ``len(estimators_)`` containing the
        class labels for each estimator in the chain.
    estimators_ : list
        A list of clones of base_estimator.
    order_ : list
        The order of labels in the classifier chain.
    See also
    --------
    RegressorChain: Equivalent for regression
    MultioutputClassifier: Classifies each output independently rather than
        chaining.
    References
    ----------
    Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank, "Classifier
    Chains for Multi-label Classification", 2009.
    """

    def fit(self, X, Y):
        """Fit the model to data matrix X and targets Y.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        Y : array-like, shape (n_samples, n_classes)
            The target values.
        Returns
        -------
        self : object
        """
        super(ClassifierChain, self).fit(X, Y)
        self.classes_ = []
        for chain_idx, estimator in enumerate(self.estimators_):
            self.classes_.append(estimator.classes_)
        return self

    @if_delegate_has_method('base_estimator')
    def predict_proba(self, X):
        """Predict probability estimates.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Returns
        -------
        Y_prob : array-like, shape (n_samples, n_classes)
        """
        X = check_array(X, accept_sparse=True)
        Y_prob_chain = []
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_prob_chain.append(estimator.predict_proba(X_aug))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)
        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_prob = [ Y_prob_chain[i] for i in inv_order]

        return Y_prob

    @if_delegate_has_method('base_estimator')
    def decision_function(self, X):
        """Evaluate the decision_function of the models in the chain.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        Returns
        -------
        Y_decision : array-like, shape (n_samples, n_classes )
            Returns the decision function of the sample for each model
            in the chain.
        """
        Y_decision_chain = []
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_decision_chain.append(estimator.decision_function(X_aug))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_decision = [ Y_decision_chain[i] for i in inv_order]

        return Y_decision
