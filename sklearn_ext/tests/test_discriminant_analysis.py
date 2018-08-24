from sklearn.utils.testing import assert_array_almost_equal

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn_ext.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn_ext.discriminant_analysis import RegularizedDiscriminantAnalysis


def test_rda():
    for n_classes in [2, 3]:
        n_features = 5
        n_samples = 1000
        X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                          centers=n_classes, random_state=11)

        # 1. LDA/RDA
        lda_clf = LinearDiscriminantAnalysis(solver='lsqr')
        rda_clf = RegularizedDiscriminantAnalysis(
            reg_param_alpha=0., reg_param_gamma=0.)

        lda_clf.fit(X, y)
        rda_clf.fit(X, y)

        assert_array_almost_equal(
            lda_clf.predict_proba(X),
            rda_clf.predict_proba(X))

        # 2. LDA/QDA
        qda_clf = QuadraticDiscriminantAnalysis()
        rda_clf = RegularizedDiscriminantAnalysis(
            reg_param_alpha=1., reg_param_gamma=0.)

        qda_clf.fit(X, y)
        rda_clf.fit(X, y)

        assert_array_almost_equal(
            qda_clf.predict_proba(X),
            rda_clf.predict_proba(X))

        # 3. QDA/GNB
        gnb_clf = GaussianNB()
        rda_clf = RegularizedDiscriminantAnalysis(
            reg_param_alpha=1., reg_param_gamma=1.)

        gnb_clf.fit(X, y)
        rda_clf.fit(X, y)

        assert_array_almost_equal(
            gnb_clf.predict_proba(X),
            rda_clf.predict_proba(X))
