import numpy as np

from sklearn_ext.wiener_filter import WienerFilter


def make_data(n_samples,
              n_features,
              n_outputs,
              n_lags,
              noise_variance,
              random_seed):
    np.random.seed(seed=random_seed)
    X = np.random.randn(n_samples, n_features)
    Y = np.zeros((n_samples, n_outputs))
    W = np.random.randn(n_features, n_lags, n_outputs)
    b = np.random.randn(n_outputs)
    for sample in range(n_lags, n_samples):
        X_tmp = X[sample - n_lags:sample, :]
        for output in range(n_outputs):
            W_tmp = W[:, :, output]
            Y[sample, output] = np.dot(X_tmp.ravel(), W_tmp.ravel()) + \
                b[output] + noise_variance * np.random.randn()

    return X, Y


def test_wiener_filter():
    n_lags = 5
    X, Y = make_data(n_samples=int(1e4),
                     n_features=10,
                     n_outputs=10,
                     n_lags=n_lags,
                     noise_variance=0.,
                     random_seed=606)
    wf = WienerFilter(n_lags=n_lags, reg_lambda=0.)
    wf.fit(X, Y)
    assert wf.score(X, Y) == 1
