import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc


__all__ = ['RocThreshold']


class RocThreshold(object):
    """Choose optimal class-specific thresholds by using ROC analysis.

    In the multi-class setting, one-vs-all classifiers are used to estimate
    the thresholds for each class.

    Parameters
    ----------
    strategy : str, default="max_random"
        Strategy to use to estimate thresholds
        * "max_random": maximizes distance from random classifier (i.e.
        distance from line x=y).
        * "min_perfect": minimizes distance from perfect classifier (i.e.
        distance from point (0,1).
        * "fpr_threshold": chooses the threshold that maximizes the true positive
        rate (TPR), subject to false positive rate (FPR) < fpr_threshold.
        * "tpr_threshold": chooses the threshold that minimizes the false positive
        rate (FPR) subject to true positive rate (TPR) > tpr_threshold.
    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds when computing ROC curve
        metrics.
    fpr_threshold : float or array of shape (n_class,), optional (default=None)
        False positive rate max limit when using "fpr_threshold" strategy.
    tpr_threshold : float or array of shape (n_class,), optional (default=None)
        True positive rate min limit when using "tpr_threshold" strategy.
    theta_max : float or array of shape (n_class,), optional (default=None)
        Maximum allowed threshold(s).
    theta_min : float or array of shape (n_class,), optional (default=None)
        Minimum allowed threshold(s).


    Attributes
    ----------
    classes_ : array of shape (n_class,)
        Holds the label for each class.
    fpr_ : array, shape (n_thresholds,)
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].
    tpr_ : array, shape (n_thresholds,)
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
    thresholds_ : array of shape (n_thresholds,)
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.
    roc_auc_ : array of shape(n_class,)
        Area under the curve metric for each class.
    theta_opt_ : array of shape (n_class,)
        Optimal thresholds for each class estimated with selected strategy.


    """

    def __init__(self,
                 strategy='max_random',
                 drop_intermediate=False,
                 fpr_threshold=None,
                 tpr_threshold=None,
                 theta_max=None,
                 theta_min=None):
        self.strategy = strategy
        self.drop_intermediate = drop_intermediate
        self.fpr_threshold = fpr_threshold
        self.tpr_threshold = tpr_threshold
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.classes_ = None
        self.thresholds_ = dict()
        self.tpr_ = dict()
        self.fpr_ = dict()
        self.roc_auc_ = dict()
        self.theta_opt_ = dict()

    def fit(self, y_true, y_pred):
        """ Compute class-specific TPR, FPR and find optimal ROC thresholds.

        Parameters
        ----------

        y_true : array-like or label indicator matrix
            Ground truth (correct) labels for n_samples samples.
        y_pred : array-like of float, shape = (n_samples, n_classes)
            Predicted probabilities, as returned by a classifier’s predict_proba method.
        """


        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
        self.classes_ = le.classes_

        for i, class_ in enumerate(self.classes_):
            idx = np.where(y_true == i)
            # Convert to one-vs-all classifier and estimate probabilities for
            # the instances belonging to the class
            y_onevsall = np.zeros_like(y_true)
            y_onevsall[idx] = 1
            y_pred_onevsall = y_pred[:, i]
            self.fpr_[class_], self.tpr_[class_], self.thresholds_[class_] = \
                roc_curve(y_onevsall,
                          y_pred_onevsall,
                          drop_intermediate=self.drop_intermediate)
            self.roc_auc_[class_] = auc(self.fpr_[class_], self.tpr_[class_])

        if self.strategy == 'max_random':
            self._compute_thresholds_max_random()
        elif self.strategy == 'min_perfect':
            self._compute_thresholds_min_perfect()
        elif self.strategy == 'fpr_threshold':
            if self.fpr_threshold is None:
                raise ValueError("fpr_threshold strategy requires the fpr_threshold "
                                 "argument to be set.")
            else:
                self._compute_thresholds_fpr_threshold()
        elif self.strategy == 'tpr_threshold':
            if self.tpr_threshold is None:
                raise ValueError("fpr_threshold strategy requires the fpr_threshold "
                                 "argument to be set.")
            else:
                self._compute_thresholds_tpr_threshold()
        else:
            raise ValueError("Unrecognized strategy for computing thresholds: "
                             "{}.".format(self.strategy))

        self._check_thresholds_limits()

    def _compute_thresholds_max_random(self):
        for c_ in self.classes_:
            rnd_clf_tpr = np.linspace(0, 1, self.thresholds_[c_].size)
            self.theta_opt_[c_] = self.thresholds_[
                c_][np.argmax(self.tpr_[c_] - rnd_clf_tpr)]

    def _compute_thresholds_min_perfect(self):
        for c_ in self.classes_:
            self.theta_opt_[c_] = self.thresholds_[c_][np.argmin(
                np.sqrt((self.tpr_[c_] - 1)**2 + (self.fpr_[c_] - 0)**2))]

    def _compute_thresholds_fpr_threshold(self):
        if isinstance(self.fpr_threshold, (list, tuple, np.ndarray)):
            fpr_threshold = self.fpr_threshold
        else:
            fpr_threshold = [self.fpr_threshold] * len(self.classes_)

        for c_ in self.classes_:
            turning_point = np.where(self.fpr_[c_] > fpr_threshold[c_])[0][0]
            self.theta_opt_[c_] = self.thresholds_[c_][turning_point - 1]

    def _compute_thresholds_tpr_threshold(self):
        if isinstance(self.tpr_threshold, (list, tuple, np.ndarray)):
            tpr_threshold = self.tpr_threshold
        else:
            tpr_threshold = [self.tpr_threshold] * len(self.classes_)

        for c_ in enumerate(self.classes_):
            turning_point = np.where(self.tpr_[c_] > tpr_threshold[c_])[0][0]
            self.theta_opt_[c_] = self.thresholds_[c_][turning_point]

    def _check_thresholds_limits(self):
        if self.theta_min is not None:
            if isinstance(self.theta_min, (list, tuple, np.ndarray)):
                min_thresholds = self.theta_min
            else:
                min_thresholds = [self.theta_min] * len(self.classes_)

            for c_ in self.classes_:
                if self.theta_opt_[c_] < min_thresholds[c_]:
                    self.theta_opt_[c_] = min_thresholds[c_]

        if self.theta_max is not None:
            if isinstance(self.theta_max, (list, tuple, np.ndarray)):
                max_thresholds = self.theta_max
            else:
                max_thresholds = [self.theta_max] * len(self.classes_)

            for c_ in self.classes_:
                if self.theta_opt_[c_] > max_thresholds[c_]:
                    self.theta_opt_[c_] = max_thresholds[c_]
