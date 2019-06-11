from sklearn.preprocessing import MinMaxScaler

__all__ = [
    'RobustMinMaxScaler'
]


class RobustMinMaxScaler(MinMaxScaler):
    """Non-linear MinMaxScaler with offset.

    This is similar to MinMaxScaler, except it applies a non-linear
    (thresholding) operation at the edges allowing for a high and a low offset
    defined by the offset argument.

    Parameters
    ----------
    desired_feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    offset : tuple (low, high), default=(0.1, 0.1)
        Desired offset for thresholding.
    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    Attributes
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Actual range of transformed data before thresholding.
    min_ : ndarray, shape (n_features,)
        Per feature adjustment for minimum.
    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.
        .. versionadded:: 0.17
           *scale_* attribute.
    data_min_ : ndarray, shape (n_features,)
        Per feature minimum seen in the data
        .. versionadded:: 0.17
           *data_min_*
    data_max_ : ndarray, shape (n_features,)
        Per feature maximum seen in the data
        .. versionadded:: 0.17
           *data_max_*
    data_range_ : ndarray, shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data
        .. versionadded:: 0.17
           *data_range_*

    Notes
    ----------
    Due to the applied non-linearity, this is an irreversible transformation.
    That is, in the general case, X is not equal to
    scaler.inverse_transform(scaler.transform(X)).

    """

    def __init__(self,
                 desired_feature_range=(0, 1),
                 offset=(0.1, 0.1),
                 copy=True):
        super(RobustMinMaxScaler, self).__init__(
            feature_range=(
                desired_feature_range[0] - offset[0],
                desired_feature_range[1] + offset[1]),
            copy=copy)
        self.desired_feature_range = desired_feature_range
        self.offset = offset

    def transform(self, x):
        x_sc = super(RobustMinMaxScaler, self).transform(x)
        x_sc[x_sc < self.feature_range[0] + self.offset[0]] = \
            self.feature_range[0] + self.offset[0]
        x_sc[x_sc > self.feature_range[1] - self.offset[1]] = \
            self.feature_range[1] - self.offset[1]
        return x_sc
