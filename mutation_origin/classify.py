from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ShuffleSplit, GridSearchCV

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.3"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


def one_class_svm(feat, seed):
    """one-class classification using SVM"""
    svm = OneClassSVM(nu=0.3, kernel="linear", random_state=seed)
    svm.fit(feat)
    return svm


def logistic_regression(feat, resp, seed, c_values, penalty_options, n_jobs):
    """fits a logistic regression classifier using a grid search with CV"""
    param_grid = {'C': c_values, 'penalty': penalty_options}
    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    log_reg = LogisticRegression(class_weight='balanced')
    classifier = GridSearchCV(estimator=log_reg, param_grid=param_grid,
                              scoring='roc_auc',
                              cv=shuffle_split,
                              n_jobs=n_jobs)
    classifier.fit(feat, resp)
    return classifier


def naive_bayes(feat, resp, seed, alphas, n_jobs):
    """fits a logistic regression classifier using a grid search with CV"""
    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    param_grid = {'alpha': alphas}
    nb = BernoulliNB()
    classifier = GridSearchCV(estimator=nb, param_grid=param_grid,
                              scoring='roc_auc',
                              cv=shuffle_split,
                              n_jobs=n_jobs)
    classifier.fit(feat, resp)
    return classifier


def predict_origin(classifier, features):
    predictions = classifier.predict(features)
    scores = classifier.decision_function(features)
    return predictions, scores
