from __future__ import division

import warnings
import numpy as np

from sklearn import metrics
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.validation import _num_samples


__all__ = [
    'accuracy_score',
    'hamming_score',
    'zero_one_loss',
    'hamming_loss',
    'multiclass_multioutput'
]


def _check_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task
    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.
    Column vectors are squeezed to 1d, while multilabel formats are returned
    as CSR sparse label indicators.
    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary',
                        multiclass-multioutput}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``
    y_true : array or indicator matrix
    y_pred : array or indicator matrix
    """
    check_consistent_length(y_true, y_pred)
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    y_type = set([type_true, type_pred])
    if y_type == set(["binary", "multiclass"]):
        y_type = set(["multiclass"])

    if len(y_type) > 1:
        raise ValueError("Classification metrics can't handle a mix of {0} "
                         "and {1} targets".format(type_true, type_pred))

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    if (y_type not in ["binary", "multiclass", "multilabel-indicator",
                       "multiclass-multioutput"]):
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        y_true = column_or_1d(y_true)
        y_pred = column_or_1d(y_pred)
        if y_type == "binary":
            unique_values = np.union1d(y_true, y_pred)
            if len(unique_values) > 2:
                y_type = "multiclass"

    if y_type.startswith('multilabel'):
        y_type = 'multilabel-indicator'

    return y_type, y_true, y_pred


def _weighted_sum(sample_score, weight, normalize=False):
    """Computes a weighted sum"""
    if normalize:
        # np.average can handle the case where weight=None
        return np.average(sample_score, weights=weight)
    elif weight is not None:
        return np.dot(sample_score, weight)
    else:
        return sample_score.sum()


def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Read more in the :ref:`User Guide <accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    normalize : bool, optional (default=True)
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    score : float
        If ``normalize == True``, return the correctly classified samples
        (float), else it returns the number of correctly classified samples
        (int).
        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.
    See also
    --------
    hamming_score, hamming_loss, zero_one_loss
    Notes
    -----
    In binary and multiclass classification, this function is equal
    to the ``hamming_score`` function.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn_ext.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred)
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False)
    2
    In the multilabel case with binary label indicators:
    >>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    0.5
    """

    # Compute accuracy for each possible representation
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)
    if y_type.startswith(('multilabel', 'multiclass-multioutput')):
        differing_labels = np.count_nonzero(y_true - y_pred, axis=1)
        score = differing_labels == 0
    else:
        score = y_true == y_pred

    return _weighted_sum(score, sample_weight, normalize)


def zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None):
    """Zero-one classification loss.
    If normalize is ``True``, return the fraction of misclassifications
    (float), else it returns the number of misclassifications (int). The best
    performance is 0.
    Read more in the :ref:`User Guide <zero_one_loss>`.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    normalize : bool, optional (default=True)
        If ``False``, return the number of misclassifications.
        Otherwise, return the fraction of misclassifications.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    loss : float or int,
        If ``normalize == True``, return the fraction of misclassifications
        (float), else it returns the number of misclassifications (int).
    Notes
    -----
    In multilabel classification, the zero_one_loss function corresponds to
    the subset zero-one loss: for each sample, the entire set of labels must be
    correctly predicted, otherwise the loss for that sample is equal to one.
    See also
    --------
    accuracy_score, hamming_loss, jaccard_similarity_score
    Examples
    --------
    >>> from sklearn_ext.metrics import zero_one_loss
    >>> y_pred = [1, 2, 3, 4]
    >>> y_true = [2, 2, 3, 4]
    >>> zero_one_loss(y_true, y_pred)
    0.25
    >>> zero_one_loss(y_true, y_pred, normalize=False)
    1
    In the multilabel case with binary label indicators:
    >>> zero_one_loss(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    0.5
    """
    score = accuracy_score(y_true, y_pred,
                           normalize=normalize,
                           sample_weight=sample_weight)

    if normalize:
        return 1 - score
    else:
        if sample_weight is not None:
            n_samples = np.sum(sample_weight)
        else:
            n_samples = _num_samples(y_true)
        return n_samples - score


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    """Hamming classification score.
    In multilabel classification, this function computes label-based accuracy.
    Read more in the :ref:`User Guide <accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    labels : array, shape = [n_labels], optional (default=None)
        Integer array of labels. If not provided, labels will be inferred
        from y_true and y_pred.
        .. versionadded:: 0.18
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    score : float
        If ``normalize == True``, return the correctly classified samples
        (float), else it returns the number of correctly classified samples
        (int).
        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.
    See also
    --------
    accuracy_score, hamming_loss, zero_one_loss
    Notes
    -----
    In binary and multiclass classification, this function is equal
    to the ``accuracy_score`` function.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn_ext.metrics import hamming_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> hamming_score(y_true, y_pred)
    0.5
    >>> hamming_score(y_true, y_pred, normalize=False)
    2
    In the multilabel case with binary label indicators:
    >>> hamming_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    0.75
    """

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)
    score = y_true == y_pred
    return _weighted_sum(score, sample_weight, normalize)


def hamming_loss(y_true, y_pred, normalize=True, sample_weight=None):
    """Compute the average Hamming loss.
    The Hamming loss is the fraction of labels that are incorrectly predicted.
    Read more in the :ref:`User Guide <hamming_loss>`.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    labels : array, shape = [n_labels], optional (default=None)
        Integer array of labels. If not provided, labels will be inferred
        from y_true and y_pred.
        .. versionadded:: 0.18
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
        .. versionadded:: 0.18
    Returns
    -------
    loss : float or int,
        Return the average Hamming loss between element of ``y_true`` and
        ``y_pred``.
    See Also
    --------
    accuracy_score, jaccard_similarity_score, zero_one_loss
    Notes
    -----
    In multiclass classification, the Hamming loss correspond to the Hamming
    distance between ``y_true`` and ``y_pred`` which is equivalent to the
    subset ``zero_one_loss`` function.
    In multilabel classification, the Hamming loss is different from the
    subset zero-one loss. The zero-one loss considers the entire set of labels
    for a given sample incorrect if it does entirely match the true set of
    labels. Hamming loss is more forgiving in that it penalizes the individual
    labels.
    The Hamming loss is upperbounded by the subset zero-one loss. When
    normalized over samples, the Hamming loss is always between 0 and 1.
    References
    ----------
    .. [1] Grigorios Tsoumakas, Ioannis Katakis. Multi-Label Classification:
           An Overview. International Journal of Data Warehousing & Mining,
           3(3), 1-13, July-September 2007.
    .. [2] `Wikipedia entry on the Hamming distance
           <https://en.wikipedia.org/wiki/Hamming_distance>`_
    Examples
    --------
    >>> from sklearn_ext.metrics import hamming_loss
    >>> y_pred = [1, 2, 3, 4]
    >>> y_true = [2, 2, 3, 4]
    >>> hamming_loss(y_true, y_pred)
    0.25
    In the multilabel case with binary label indicators:
    >>> hamming_loss(np.array([[0, 1], [1, 1]]), np.zeros((2, 2)))
    0.75
    """

    score = hamming_score(y_true, y_pred,
                          normalize=normalize,
                          sample_weight=sample_weight)

    if normalize:
        return 1 - score
    else:
        if sample_weight is not None:
            n_samples = np.sum(sample_weight)
        else:
            n_samples = _num_samples(y_true)
        return n_samples - score


def multiclass_multioutput(metric, y_true, y_pred, normalize=True,
                           sample_weight=None, class_average='binary',
                           output_weight=None, output_normalize=True):
    """Extends classification metrics to support multiclass and multilabel

    The metric is calculated for each output independently (i.e. the metric
    function is used within its multiclass support). An output-average
    (i.e. macro-average) is then computed optionally using a different weight
    for each label.

    As a consequence of the above, metrics which are strict in the multilabel
    setting, such as accuracy_score and zero_one_loss lose this property when
    converted to multiclass_multioutput metrics.

    For metrics not supporting multi-class out of the box, such as
    `precision_score`, `recall_score` and `f1_score`, a class-averaging
    strategy needs to be defined.

    Parameters
    ----------
    metric : string
        One of {'accuracy_score', 'zero_one_loss', 'log_loss',
        'jaccard_similarity_score', 'precision_score', 'recall_score',
        'f1_score'}
    y_true : array-like
    y_pred : array-like
    normalize : bool, optional (default=True)
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    class_average : string, [None, 'binary' (default), 'micro', 'macro', \
                             'samples', 'weighted']
        This parameter is required for multiclass targets.
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`)
    output_weight : array-like of shape = [n_outputs], optional
        Output weights.
    output_normalize : bool, optional (default=True)
        If ``False``, return the sum of the computed scores for each label.
        Otherwise, return their average.

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``
    y_true : array or indicator matrix
    y_pred : array or indicator matrix
    """
    if metric not in ['accuracy_score', 'zero_one_loss', 'log_loss',
                      'hamming_loss', 'jaccard_similarity_score',
                      'precision_score', 'recall_score', 'f1_score']:
        raise ValueError("{} score not supported.".format(metric))

    score_function = getattr(metrics, metric)
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if y_true.ndim == 1:
        msg = ("Targets are 1-D. Assuming multioutput targets and a single "
               "sample.")
        warnings.warn(msg)
        y_true = np.reshape(y_true, (1, -1))
        y_pred = np.reshape(y_pred, (1, -1))

    num_samples, num_outputs = y_true.shape
    if output_weight is not None and len(output_weight) is not num_outputs:
        raise ValueError("""The length of the weight vector must equal the
                         number of outputs.""")

    scores = np.zeros((num_outputs,))
    for output in range(num_outputs):
        if score_function in [metrics.accuracy_score, metrics.zero_one_loss,
                              metrics.log_loss,
                              metrics.jaccard_similarity_score]:
            scores[output] = score_function(
                y_true=y_true[:, output].reshape(num_samples, -1),
                y_pred=y_pred[:, output].reshape(num_samples, -1),
                normalize=normalize, sample_weight=sample_weight)

        # hamming_loss does not support normalize as of version 0.19
        if score_function is metrics.hamming_loss:
            scores[output] = score_function(
                y_true=y_true[:, output].reshape(num_samples, -1),
                y_pred=y_pred[:, output].reshape(num_samples, -1),
                sample_weight=sample_weight)

        # the following metrics do not support multi-class out of the box
        # so averaging is needed
        elif score_function in [metrics.precision_score, metrics.recall_score,
                                metrics.f1_score]:
            scores[output] = score_function(
                y_true=y_true[:, output].reshape(num_samples, -1),
                y_pred=y_pred[:, output].reshape(num_samples, -1),
                average=class_average, sample_weight=sample_weight)
    return _weighted_sum(scores, output_weight, output_normalize)
