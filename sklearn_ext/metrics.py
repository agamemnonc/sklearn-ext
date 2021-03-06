from itertools import chain

import warnings
import numpy as np

from sklearn import metrics
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.multiclass import _unique_multiclass, _unique_indicator
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from sklearn.utils.validation import _num_samples


__all__ = [
    'accuracy_score',
    'hamming_loss',
    'hamming_score',
    'multiclass_multioutput',
    'zero_one_loss',
    'median_absolute_error'
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

    if y_type == set(["multiclass-multioutput", "multilabel-indicator"]):
        y_type = set(["multiclass-multioutput"])

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


_FN_UNIQUE_LABELS = {
    'binary': _unique_multiclass,
    'multiclass': _unique_multiclass,
    'multiclass-multioutput': _unique_multiclass,
    'multilabel-indicator': _unique_indicator,
}


def unique_labels(*ys):
    """Extract an ordered array of unique labels

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes

    Returns
    -------
    out : numpy array of shape [n_unique_labels]
        An ordered array of unique labels.

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    """
    if not ys:
        raise ValueError('No argument has been passed.')
    # Check that we don't mix label format

    ys_types = set(type_of_target(x) for x in ys)
    if ys_types == set(["binary", "multiclass"]):
        ys_types = set(["multiclass"])

    if ys_types == set(['multiclass-multioutput', 'multilabel-indicator']):
        ys_types = set(['multiclass-multioutput'])

    if len(ys_types) > 1:
        raise ValueError("Mix type of y not allowed, got types %s" % ys_types)

    label_type = ys_types.pop()

    # Check consistency for the indicator format
    if (label_type == "multilabel-indicator" and
            len(set(check_array(y, ['csr', 'csc', 'coo']).shape[1]
                    for y in ys)) > 1):
        raise ValueError("Multi-label binary indicator input with "
                         "different numbers of labels")

    # Get the unique set of labels
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
    if not _unique_labels:
        raise ValueError("Unknown label type: %s" % repr(ys))

    ys_labels = set(chain.from_iterable(_unique_labels(y) for y in ys))

    # Check that we don't mix string type with number type
    if (len(set(isinstance(label, str) for label in ys_labels)) > 1):
        raise ValueError("Mix of label input types (string and number)")

    return np.array(sorted(ys_labels))


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


def normalize_confusion_matrix(C):
    """Normalizes the rows of a confusion matrix so that their sum is equal to
    one.

    Parameters
    ----------
    C : array, shape = [n_classes, n_classes]
        Confusion matrix

    Returns
    -------
    Cnorm : array, shape = [n_classes, n_classes]
        Normalized confusion matrix
    """
    return C / C.sum(axis=1)[:, np.newaxis]


def multiclass_multioutput(y_true, y_pred, metric, output_average='macro',
                           output_normalize=True, output_weight=None,
                           **kwargs):
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

    A confusion matrix makes

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels, as returned by a classifier.

    metric : str, ['accuracy_score', 'zero_one_loss', 'log_loss',
        'jaccard_similarity_score', 'precision_score', 'recall_score',
        'f1_score', 'confusion_matrix']

    output_average : str, [None, 'macro' (default)]
        Output average strategy.
        If ``None``, the scores for each output are
        returned. If ``macro``, an output macro-average is returned.

    output_weight : array-like of shape = [n_outputs], optional
        Output weights. Only valid when output_average is set to ``macro``.

    output_normalize : bool, optional (default=True)
        If ``False``, return the sum of the computed scores for each label.
        Otherwise, return their average. Only valid when output_average is set
        to ``macro``.

    **kwargs : additional arguments
        Additional parameters to be passed to score function.

    Returns
    -------
    score : float or array of float, shape = [n_classes,]
        Specified score.

    Notes
    -------
    Other types of averaging (e.g. micro, variance_weighted) are not currently
    supported, since it is not clear how they would make sense in a
    multi-output scenario.
    """
    if metric not in ['accuracy_score', 'zero_one_loss', 'log_loss',
                      'hamming_loss', 'jaccard_similarity_score',
                      'precision_score', 'recall_score', 'f1_score',
                      'confusion_matrix']:
        raise ValueError("{} score not supported.".format(metric))

    score_function = getattr(metrics, metric)
    if metric == 'log_loss':
        for y_output in y_pred:
            y_output = check_array(y_output, ensure_2d=False)
    else:
        y_type, y_true, y_pred = _check_targets(y_true, y_pred)
        if y_true.ndim == 1:
            msg = (
                "Targets are 1-D. Assuming multioutput targets and a single "
                "sample.")
            warnings.warn(msg)
            y_true = np.reshape(y_true, (1, -1))
            y_pred = np.reshape(y_pred, (1, -1))

    n_samples, n_outputs = y_true.shape

    if output_weight is not None and len(output_weight) is not n_outputs:
        raise ValueError("""The length of the weight vector must equal the
                         number of outputs.""")

    # infer the labels if they are not provided
    if metric in ['precision_score', 'recall_score', 'f1_score',
                  'confusion_matrix']:
        present_labels = unique_labels(y_true, y_pred)
        if 'labels' in kwargs:
            labels = kwargs.get('labels')
        else:
            labels = present_labels
        n_labels = len(labels)

    scores = []
    for output in range(n_outputs):
        y_true_output = y_true[:, output].reshape(n_samples, -1)
        if metric == 'log_loss':
            y_pred_output = y_pred[output]
        else:
            y_pred_output = y_pred[:, output].reshape(n_samples, -1)

        scores.append(score_function(y_true_output, y_pred_output, **kwargs))

    scores = np.array(scores)
    if output_average is None:
        return scores
    elif output_average == 'macro':
        if metric in ['accuracy_score', 'zero_one_loss',
                      'jaccard_similarity_score', 'hamming_loss', 'log_loss']:
            return _weighted_sum(scores, output_weight, output_normalize)
        elif metric in ['precision_score', 'recall_score', 'f1_score']:
            if kwargs.get('average') is None:
                # return one output-average for each label
                avg_scores = np.zeros((n_labels,))
                for i in range(n_labels):
                    avg_scores[i] = _weighted_sum(
                        scores[:, i], output_weight, output_normalize)

                return avg_scores
            else:
                return _weighted_sum(scores, output_weight, output_normalize)
        elif metric == 'confusion_matrix':
            avg_cm = np.zeros((n_labels, n_labels))
            for i in range(n_labels):
                for j in range(n_labels):
                    avg_cm[i, j] = _weighted_sum(
                        scores[:, i, j], output_weight, output_normalize)

            return avg_cm
    else:
        raise ValueError("Average setting not supported.")

def median_absolute_error(y_true, y_pred, multioutput='uniform_average'):
    """Median absolute error regression loss
    Median absolute error output is non-negative floating point. The best value
    is 0.0. Read more in the :ref:`User Guide <median_absolute_error>`.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape
        (n_outputs,)
        Defines aggregating of multiple output values. Array-like value defines
        weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.
    Examples
    --------
    >>> from sklearn.metrics import median_absolute_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> median_absolute_error(y_true, y_pred)
    0.5
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> median_absolute_error(y_true, y_pred)
    0.75
    >>> median_absolute_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> median_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.85
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    output_errors = np.median(np.abs(y_pred - y_true), axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)

def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    """Check that y_true and y_pred belong to the same regression task
    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().
    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'
    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.
    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    dtype: str or list, default="numeric"
        the dtype argument passed to check_array
    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("y_true and y_pred have different number of output "
                         "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ('raw_values', 'uniform_average',
                               'variance_weighted')
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError("Allowed 'multioutput' string values are {}. "
                             "You provided multioutput={!r}".format(
                                 allowed_multioutput_str,
                                 multioutput))
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in "
                             "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(("There must be equally many custom weights "
                              "(%d) as outputs (%d).") %
                             (len(multioutput), n_outputs))
    y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

    return y_type, y_true, y_pred, multioutput
