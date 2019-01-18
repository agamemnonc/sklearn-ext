"""
ClassifierChain predict_proba and decision_function fixes
"""

import numpy as np
import scipy.sparse as sp
from scipy.stats import mode
from sklearn.base import MetaEstimatorMixin, ClassifierMixin, TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.multioutput import _BaseChain
from sklearn.ensemble.voting_classifier import _parallel_fit_estimator

from sklearn.utils.metaestimators import if_delegate_has_method

__all__ = ['ClassifierChain',
           'ExtendedClassifierChain']

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

class ExtendedClassifierChain(_BaseComposition, ClassifierMixin,
                              TransformerMixin):
    """Soft Voting/Majority Rule classifier for unfitted classifier chains.
    Parameters
    ----------
    base_estimator : estimator
        The base estimator from which the classifier chains are built.
        Invoking the ``fit`` method on the ``ExtendedClassifierChain`` will fit
        clones of the original estimators that will be stored in the class
        attribute ``self.estimators_``.
    n_chains : int
        Number of classifier chains to be used.
    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.
    orders : array-like, shape=[n_chains, n_outputs] or 'random', optional
        By default a random ordering will be used for each chain.
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
    weights : array-like, shape = [n_chains], optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Attributes
    ----------
    chains_ : list of classifier chains
        The collection of fitted classifier chains.
    named_estimators_ : Bunch object, a dictionary with attribute access
        Attribute to access any fitted sub-estimators by name.
    classes_ : array-like, shape = [n_predictions]
        The classes labels.
    """

    def __init__(self, base_estimator, n_chains=10, voting='hard',
                  orders=None, cv=None, random_state=None, weights=None,
                  n_jobs=None):
        self.base_estimator = base_estimator
        self.n_chains = n_chains
        self.voting = voting
        self.orders = orders
        self.cv = cv
        self.random_state = random_state
        self.weights = weights
        self.n_jobs = n_jobs

    def fit(self, X, Y, sample_weight=None):
        """ Fit the estimators.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.
        Returns
        -------
        self : object
        """
        X, Y = check_X_y(X, Y, multi_output=True, accept_sparse=True)

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if (self.weights is not None and
                len(self.weights) != self.n_chains):
            raise ValueError("Number of classifiers and weights must be equal"
                             '; got %d weights, %d chains'
                             % (len(self.weights), len(self.n_chains)))

        if sample_weight is not None:
            if not has_fit_parameter(self.base_estimator, 'sample_weight'):
                raise ValueError("Underlying estimator \'%s\' does not"
                                 " support sample weights." %
                                 self.base_estimator.__class__.__name__)

        random_state = check_random_state(self.random_state)

        if self.orders is not None:
            if np.asarray(self.orders).shape != (self.n_chains, Y.shape[1]):
                raise ValueError("Argument orders must have shape " + \
                                 "(n_chains, n_outputs); expected {}, " + \
                                 "but got {}.".format((self.n_chains, 
                                          Y.shape[1]), self.orders.shape))
            else:
                self.orders_ = self.orders
        else:
            self.orders_ = [random_state.permutation(Y.shape[1]) for _ in \
                            range(self.n_chains)]

        self.le_ = []
        self.classes_ = []
        for y in Y.T:
            le = LabelEncoder().fit(y)
            self.le_.append(le)
            self.classes_.append(le.classes_)
            
        self.chains_ = [ClassifierChain(self.base_estimator,
                                        order=order,
                                        cv=self.cv) for order in self.orders_]
        self.chains_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(cc), X, Y,
                        sample_weight) for cc in self.chains_)

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Returns
        ----------
        maj : array-like, shape (n_samples, n_classes)
            Predicted class labels.
        """
        for chain in self.chains_:
            for estimator in chain.estimators_:
                check_is_fitted(estimator, 'coef_')
        
        if self.voting == 'soft':
            proba = self.predict_proba(X)
            maj = np.zeros((X.shape[0], len(proba)))
            for output, proba_output in enumerate(proba):
                maj[:, output] = self.le_[output].inverse_transform(
                        np.argmax(proba_output, axis=1))

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.zeros((X.shape[0], len(predictions)))
            for i, one_prediction in enumerate(predictions):
                maj[:, i] = mode(one_prediction, axis=1)[0].squeeze()

        return maj

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Input samples.
        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        #check_is_fitted(self, 'estimators_')
        avg = list(np.average(self._collect_probas(X), axis=0,
                         weights=self._weights_not_none))
        return avg
    
    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.chains_])

    def _predict(self, X):  
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.chains_]).T
    
    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators"""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators,
                                    self.weights) if est[1] is not None]