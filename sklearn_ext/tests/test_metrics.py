import numpy as np

from sklearn.utils.testing import assert_almost_equal

from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn_ext.metrics import accuracy_score, zero_one_loss
from sklearn_ext.metrics import hamming_loss, hamming_score
from sklearn_ext.metrics import multiclass_multioutput

random_state = 606

# Data
Y1 = np.array([[1, 0],
               [2, 2],
               [0, 1],
               [0, 2],
               [0, 1]])

Y2 = np.array([[1, 1],
               [2, 2],
               [0, 1],
               [0, 2],
               [0, 1]])

# Multi-class multi-label data
X, Y = make_multilabel_classification(n_samples=1000,
                                      n_features=30,
                                      n_classes=5,
                                      random_state=random_state)
Y = np.column_stack((Y, Y[:, 0] + Y[:, 1], Y[:, 3] + Y[:, 2]))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=random_state)
clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
Y_true = Y_test


def test_accuracy_score():
    assert_almost_equal(accuracy_score(Y1, Y2), 0.8)


def test_zero_one_loss():
    assert_almost_equal(zero_one_loss(Y1, Y2), 0.2)


def test_hamming_score():
    assert_almost_equal(hamming_score(Y1, Y2), 0.9)


def test_hamming_loss():
    assert_almost_equal(hamming_loss(Y1, Y2), 0.1)
    assert_almost_equal(multiclass_multioutput('hamming_loss', Y1, Y2),
                        hamming_loss(Y1, Y2))


def test_accuracy_zero_one_loss():
    assert_almost_equal(accuracy_score(Y_true, Y_pred) +
                        zero_one_loss(Y_true, Y_pred), 1)


def test_hamming_loss_score():
    assert_almost_equal(hamming_score(Y_true, Y_pred) +
                        hamming_loss(Y_true, Y_pred), 1)

def test_log_loss():
    assert_almost_equal(multiclass_multioutput('log_loss', Y_true, clf.predict_proba(X_test)),
                        0.6454764507804741)