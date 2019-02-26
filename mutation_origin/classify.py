from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from xgboost_tuner.tuner import tune_xgb_params
from xgboost import XGBClassifier


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


def logistic_regression(feat, resp, seed, scoring, c_values, penalty_options, n_jobs):
    """fits a logistic regression classifier using a grid search with CV"""
    param_grid = {'C': c_values, 'penalty': penalty_options}
    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    log_reg = LogisticRegression(class_weight='balanced', solver='liblinear',
                                 max_iter=1000)
    classifier = GridSearchCV(estimator=log_reg, param_grid=param_grid,
                              scoring=scoring,
                              cv=shuffle_split,
                              n_jobs=n_jobs)
    classifier.fit(feat, resp)
    return classifier


def naive_bayes(feat, resp, seed, alphas, scoring, class_prior=None, n_jobs=1):
    """fits a logistic regression classifier using a grid search with CV"""
    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    param_grid = {'alpha': alphas}
    nb = BernoulliNB(fit_prior=class_prior is None, class_prior=class_prior)
    classifier = GridSearchCV(estimator=nb, param_grid=param_grid,
                              scoring=scoring,
                              cv=shuffle_split,
                              n_jobs=n_jobs)
    classifier.fit(feat, resp)
    return classifier


def predict_origin(classifier, features):
    predictions = classifier.predict(features)
    scores = classifier.decision_function(features)
    return predictions, scores


def xgboost(feat, resp, seed, strategy, n_jobs, verbose):
    """uses xgb tuner to conduct parameter search"""
    silent = True if not verbose else False
    rand_search_kwargs = dict(
        cv_folds=3,
        label=resp,
        metric_sklearn='accuracy',
        metric_xgb='error',
        n_jobs=n_jobs,
        objective='binary:logistic',
        random_state=seed,
        strategy='randomized',
        train=feat,
        colsample_bytree_loc=0.5,
        colsample_bytree_scale=0.2,
        subsample_loc=0.5,
        subsample_scale=0.2)
    incr_search_kwargs = dict(
        cv_folds=3,
        label=resp,
        metric_sklearn='accuracy',
        metric_xgb='error',
        n_jobs=n_jobs,
        objective='binary:logistic',
        random_state=seed,
        strategy='incremental',
        train=feat,
        colsample_bytree_min=0.8,
        colsample_bytree_max=1.0,
        subsample_min=0.8,
        subsample_max=1.0)
    kwargs = {'incremental': incr_search_kwargs,
              'randomized': rand_search_kwargs}[strategy]
    best_params, history = tune_xgb_params(verbosity_level=0, **kwargs)
    booster = XGBClassifier(**best_params)
    booster.fit(feat, resp)
    return booster
