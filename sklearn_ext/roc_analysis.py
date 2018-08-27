import numpy as np

from sklearn.metrics import roc_curve, auc


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
        point (0,1).
        * "fpr_limit": chooses the threshold that maximizes the true positive
        rate (TPR), subject to false positive rate (FPR) < fpr_limit.
        * "tpr_limit": chooses the threshold that minimizes the false positive
        rate (FPR) subject to true positive rate (TPR) > tpr_limit.
    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds when computing ROC curve
        metrics.
    fpr_max : float or array of shape [n_classes], optional (default=None)
        False positive rate max limit when using "fpr_limit" strategy.
    tpr_min : float or array of shape [n_classes], optional (default=None)
        True positive rate min limit when using "tpr_limit" strategy.
    theta_max : float or array of shape [n_classes], optional (default=None)
        Maximum allowed threshold(s).
    theta_min : float or array of shape [n_classes], optional (default=None)
        Minimum allowed threshold(s).


    Attributes
    ----------
    """

    def __init__(self,
                 strategy='max_random',
                 drop_intermediate=False,
                 fpr_max=None,
                 tpr_min=None,
                 theta_max=None,
                 theta_min=None):
        self.strategy = strategy
        self.drop_intermediate = drop_intermediate
        self.fpr_max = fpr_max
        self.tpr_min = tpr_min
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.classes_ = None
        self.thresholds_ = dict()
        self.tpr_ = dict()
        self.fpr_ = dict()
        self.roc_auc_ = dict()
        self.optimal_threshold_ = dict()

    def fit(self, y_true, y_pred):
        """ Compute class-specific TPR, FPR and find optimal ROC thresholds.

        Parameters
        ----------

        y_true : array-like or label indicator matrix
            Ground truth (correct) labels for n_samples samples.
        y_pred : array-like of float, shape = (n_samples, n_classes)
            Predicted probabilities, as returned by a classifier’s predict_proba method.
        """

        self.classes_ = np.unique(y_true)
        for i, class_ in enumerate(self.classes_):
            idx = np.where(y_true == class_)
            # Convert to one-vs-all classifier
            y_onevsall = np.zeros_like(y_true)
            y_onevsall[idx] = 1
            # Estimated probabilities for the instances belonging to the class
            y_pred_onevsall = y_pred[:, i]
            self.fpr_[i], self.tpr_[i], self.thresholds_[i] = \
                roc_curve(y_onevsall,
                          y_pred_onevsall,
                          drop_intermediate=self.drop_intermediate)
            self.roc_auc_[i] = auc(self.fpr_[i], self.tpr_[i])

        if self.strategy == 'max_random':
            self._compute_thresholds_max_random()
        elif self.strategy == 'min_perfect':
            self._compute_thresholds_min_perfect()
        elif self.strategy == 'fpr_limit':
            self._compute_thresholds_fpr_limit()
        elif self.strategy == 'tpr_limit':
            self.compute_thresholds_tpr_limit()
        else:
            raise ValueError("Unrecognized strategy for computing thresholds: "
                             "{}.".format(self.strategy))

        self._check_thresholds_limits()

    def _compute_thresholds_max_random(self):
        for i, class_ in enumerate(self.classes_):
            rnd_clf_tpr = np.linspace(0, 1, self.thresholds_[i].size)
            self.optimal_threshold_[i] = self.thresholds_[
                i][np.argmax(self.tpr_[i] - rnd_clf_tpr)]

    def _compute_thresholds_min_perfect(self):
        for i, class_ in enumerate(self.classes_):
            self.optimal_threshold_[i] = self.thresholds_[i][np.argmin(
                np.sqrt((self.tpr_[i] - 1)**2 + (self.fpr_[i] - 0)**2))]

    def _compute_thresholds_fpr_limit(self):
        for i, class_ in enumerate(self.classes_):
            turning_point = np.where(self.fpr_[i] > self.fpr_threshold)[0][0]
            class_threshold = self.thresholds_[i][turning_point]

            if self.max_threshold is not None:
                if class_threshold < self.max_threshold:
                    self.optimal_threshold_[i] = class_threshold
                else:
                    self.optimal_threshold_[i] = self.max_threshold

    def _compute_thresholds_tpr_limit(self):
        for i, class_ in enumerate(self.classes_):
            turning_point = np.where(self.tpr_[i] < self.tpr_min)[0][0]
            class_threshold = self.thresholds_[i][turning_point]

            if self.min_threshold is not None:
                if class_threshold > self.min_threshold:
                    self.optimal_threshold_[i] = class_threshold
                else:
                    self.optimal_threshold_[i] = self.min_threshold

    def _check_thresholds_limits(self):
        if self.min_threshold is not None:
            if isinstance(self.min_threshold, (list, tuple, np.ndarray)):
                min_thresholds = self.min_threshold
            else:
                min_thresholds = [self.min_threshold] * len(self.classes_)

            for i, class_ in enumerate(self.classes_):
                if self.optimal_threshold_[i] < min_thresholds[i]:
                    self.optimal_threshold_[i] = min_thresholds[i]

        if self.max_threshold is not None:
            if isinstance(self.max_threshold, (list, tuple, np.ndarray)):
                max_thresholds = self.max_threshold
            else:
                max_thresholds = [self.max_threshold] * len(self.classes_)

            for i, class_ in enumerate(self.classes_):
                if self.optimal_threshold_[i] > max_thresholds[i]:
                    self.optimal_threshold_[i] = max_thresholds[i]
