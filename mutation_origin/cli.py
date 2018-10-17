"""command line interface for mutation_origin"""
import os
import time
import pickle
import click
import pandas
from numpy.random import seed as np_seed
from scitrack import CachingLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support)

from mutation_origin.opt import (_seed, _feature_dim, _enu_path,
                                 _germline_path, _output_path, _flank_size,
                                 _train_size, _test_size, _enu_ratio,
                                 _numreps, _label_col, _proximal, _usegc,
                                 _training_path, _c_values, _penalty_options,
                                 _n_jobs, _classifier_path, _data_path,
                                 _predictions_path, _alpha_options,
                                 _overwrite)
from mutation_origin.preprocess import data_to_numeric
from mutation_origin.encoder import (
    get_scaler, inverse_transform_response, transform_response)
from mutation_origin.classify import (logistic_regression, one_class_svm,
                                      predict_origin, naive_bayes)
from mutation_origin.util import (dump_json, load_predictions, get_basename,
                                  get_classifier_label)

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.1"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


LOGGER = CachingLogger()


@click.group()
def main():
    """mutori -- for building and applying classifiers of mutation origin"""
    pass


@main.command()
@_seed
@_enu_path
@_germline_path
@_output_path
@_train_size
@_test_size
@_enu_ratio
@_numreps
@_overwrite
def sample_data(enu_path, germline_path, output_path, seed,
                train_size, test_size,
                enu_ratio, numreps, overwrite):
    """creates train/test sample data"""
    if seed is None:
        seed = int(time.time())
    LOGGER.log_args()
    start_time = time.time()
    os.makedirs(output_path, exist_ok=True)
    logfile_path = os.path.join(output_path, "logs/data_sampling.log")
    if os.path.exists(logfile_path) and not overwrite:
        click.secho(f"Exists: {logfile_path}! use overwrite to force.",
                    fg='red')
        exit(-1)

    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(enu_path)
    LOGGER.input_file(germline_path)

    enu = pandas.read_csv(enu_path, sep="\t", header=0)
    germline = pandas.read_csv(germline_path, sep="\t", header=0)
    if enu_ratio != 1:
        # adjust the test sizes for ENU/germline
        # first convert both
        germ_test_size = test_size // enu_ratio
    else:
        germ_test_size = test_size // 2
    enu_test_size = test_size - germ_test_size

    # TODO make sure these siezes are <= 50% of the inputs
    if (enu.shape[0] < enu_test_size or
            germline.shape[0] < germ_test_size):
        print(f"ENU data set size: {enu.shape[0]}")
        print(f"Germline data set size: {germline.shape[0]}")
        raise ValueError("chosen test size/enu ratio not possible")

    for rep in range(numreps):
        test_outpath = os.path.join(output_path, f"test-{rep}.tsv.gz")
        train_outpath = os.path.join(output_path, f"train-{rep}.tsv.gz")
        enu_training, enu_testing = train_test_split(
            enu,
            test_size=enu_test_size,
            train_size=train_size // 2,
            random_state=seed)

        germ_training, germ_testing = train_test_split(
            germline,
            test_size=germ_test_size,
            train_size=train_size // 2,
            random_state=seed)
        if any(map(lambda x: x.shape[0] == 0,
                   [enu_training, enu_testing, germ_training, germ_testing])):
            raise RuntimeError("screw up in creating test/train set")

        # concat the data frames
        testing = pandas.concat([enu_testing, germ_testing])
        training = pandas.concat([enu_training, germ_training])
        # write out, separately, the ENU and Germline data for train and test
        testing.to_csv(test_outpath, index=False,
                       sep="\t", compression='gzip')
        training.to_csv(train_outpath, index=False,
                        sep="\t", compression='gzip')

        LOGGER.output_file(test_outpath)
        LOGGER.output_file(train_outpath)

    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")


@main.command()
@_training_path
@_output_path
@_label_col
@_seed
@_flank_size
@_feature_dim
@_proximal
@_usegc
@_c_values
@_penalty_options
@_n_jobs
@_overwrite
def lr_train(training_path, output_path, label_col, seed,
             flank_size, feature_dim, proximal,
             usegc, c_values, penalty_options, n_jobs, overwrite):
    """logistic regression training, validation, dumps optimal model"""
    if not seed:
        seed = int(time.time())

    np_seed(seed)
    LOGGER.log_args()
    os.makedirs(output_path, exist_ok=True)

    basename = get_basename(training_path)
    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-training-lr.log")
    if os.path.exists(logfile_path) and not overwrite:
        click.secho(f"Exists: {logfile_path}! use overwrite to force.",
                    fg='red')
        exit(-1)
    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(training_path)

    start_time = time.time()
    _, resp, feat, n_dims, names = data_to_numeric(training_path,
                                                   label_col, flank_size,
                                                   feature_dim, proximal,
                                                   usegc)

    if usegc:
        # we need to scale the data
        scaler = get_scaler(feat)
        feat = scaler.transform(feat)
    classifier = logistic_regression(feat, resp, seed, c_values,
                                     penalty_options.split(","), n_jobs)
    outpath = os.path.join(output_path, f"{basename}-classifier-lr.pkl")
    betas = dict(zip(names, classifier.best_estimator_.coef_.tolist()[0]))
    result = dict(classifier=classifier.best_estimator_, betas=betas)
    result['feature_params'] = dict(feature_dim=feature_dim,
                                    flank_size=flank_size, proximal=proximal,
                                    usegc=usegc)
    if usegc:
        result['scaler'] = scaler

    with open(outpath, 'wb') as clf_file:
        pickle.dump(result, clf_file)

    LOGGER.output_file(outpath)
    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")


@main.command()
@_training_path
@_output_path
@_label_col
@_seed
@_flank_size
@_feature_dim
@_proximal
@_usegc
@_alpha_options
@_n_jobs
@_overwrite
def nb_train(training_path, output_path, label_col, seed,
             flank_size, feature_dim, proximal,
             usegc, alpha_options, n_jobs, overwrite):
    """Naive Bayes training, validation, dumps optimal model"""
    if not seed:
        seed = int(time.time())

    np_seed(seed)
    LOGGER.log_args()
    os.makedirs(output_path, exist_ok=True)

    basename = get_basename(training_path)
    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-training-nb.log")
    if os.path.exists(logfile_path) and not overwrite:
        click.secho(f"Exists: {logfile_path}! use overwrite to force.",
                    fg='red')
        exit(-1)
    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(training_path)

    start_time = time.time()
    _, resp, feat, n_dims, names = data_to_numeric(training_path,
                                                   label_col, flank_size,
                                                   feature_dim, proximal,
                                                   usegc)

    if usegc:
        # we need to scale the data
        scaler = get_scaler(feat)
        feat = scaler.transform(feat)
    classifier = naive_bayes(feat, resp, seed, alpha_options, n_jobs)
    outpath = os.path.join(output_path, f"{basename}-classifier-nb.pkl")
    betas = dict(zip(names, classifier.best_estimator_.coef_.tolist()[0]))
    result = dict(classifier=classifier.best_estimator_, betas=betas)
    result['feature_params'] = dict(feature_dim=feature_dim,
                                    flank_size=flank_size, proximal=proximal,
                                    usegc=usegc)
    if usegc:
        result['scaler'] = scaler

    with open(outpath, 'wb') as clf_file:
        pickle.dump(result, clf_file)

    LOGGER.output_file(outpath)
    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")


@main.command()
@_germline_path
@_output_path
@_label_col
@_seed
@_flank_size
@_feature_dim
@_proximal
@_overwrite
def ocs_train(germline_path, output_path, label_col, seed,
              flank_size, feature_dim, proximal, overwrite):
    """one-class svm training for outlier detection"""
    if seed is None:
        seed = int(time.time())
    LOGGER.log_args()
    start_time = time.time()
    os.makedirs(output_path, exist_ok=True)

    basename = get_basename(germline_path)
    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-training-ocs.log")
    if os.path.exists(logfile_path) and not overwrite:
        click.secho(f"Exists: {logfile_path}! use overwrite to force.",
                    fg='red')
        exit(-1)

    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(germline_path)

    start_time = time.time()
    _, _, feat, n_dims, names = data_to_numeric(germline_path,
                                                label_col, flank_size,
                                                feature_dim, proximal,
                                                one_class='g')

    classifier = one_class_svm(feat, seed)
    outpath = os.path.join(output_path, f"{basename}-classifier-ocs.pkl")
    result = dict(classifier=classifier)
    result['feature_params'] = dict(feature_dim=feature_dim,
                                    flank_size=flank_size, proximal=proximal)

    with open(outpath, 'wb') as clf_file:
        pickle.dump(result, clf_file)

    LOGGER.output_file(outpath)
    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")


@main.command()
@_classifier_path
@_data_path
@_output_path
@_overwrite
def predict(classifier_path, data_path, output_path, overwrite):
    """predict labels for data"""
    LOGGER.log_args()
    with open(classifier_path, 'rb') as clf:
        classifier = pickle.load(clf)
    try:
        feature_params = classifier["feature_params"]
        scaler = classifier.get('scaler', None)
        classifier = classifier["classifier"]
    except KeyError:
        raise ValueError("pickle formatted file does not "
                         "contain classifier")

    class_label = get_classifier_label(classifier)
    basename = get_basename(classifier_path)
    os.makedirs(output_path, exist_ok=True)
    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-predict-{class_label}.log")
    if os.path.exists(logfile_path) and not overwrite:
        click.secho(f"Exists: {logfile_path}! use overwrite to force.",
                    fg='red')
        exit(-1)

    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(classifier_path)
    LOGGER.input_file(data_path)

    start_time = time.time()
    ids, resp, feat, n_dims, names = data_to_numeric(data_path,
                                                     **feature_params)
    if scaler:
        feat = scaler.transform(feat)
    predictions, scores = predict_origin(classifier, feat)
    predictions = inverse_transform_response(predictions)
    result = {}
    result['predictions'] = {'varid': ids,
                             'predicted': predictions,
                             'scores': scores.tolist()}
    result['feature_params'] = feature_params
    result['classifier_label'] = class_label
    outpath = os.path.join(
        output_path,
        f"{basename}-predicted-{class_label}.json.gz")
    dump_json(outpath, result)
    LOGGER.output_file(outpath)
    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")


# def performance -> produces summary stats on trained classifiers
# requires input data and the predicted results
@main.command()
@_training_path
@_predictions_path
@_output_path
@_overwrite
def performance(training_path, predictions_path, output_path, overwrite):
    """produce measures of classifier performance"""
    LOGGER.log_args()
    if not (training_path or predictions_path):
        click.secho("Need data sets!", fg="red")
        exit()

    basename = get_basename(training_path)
    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-performance.log")
    if os.path.exists(logfile_path) and not overwrite:
        click.secho(f"Exists: {logfile_path}! use overwrite to force.",
                    fg='red')
        exit(-1)

    LOGGER.log_file_path = logfile_path

    LOGGER.input_file(training_path)
    LOGGER.input_file(predictions_path)

    orig = pandas.read_csv(training_path, sep="\t")
    orig = orig[['varid', 'response']]
    orig.set_index('varid', inplace=True)
    predicted, feature_params = load_predictions(predictions_path)
    predicted.set_index('varid', inplace=True)
    new = orig.join(predicted)

    response_labels = inverse_transform_response([-1, 1])
    expect = transform_response(new['response'])
    predict = transform_response(new['predicted'])
    pr, re, fs, sup = precision_recall_fscore_support(expect,
                                                      predict)
    result = {}
    result['classification_report'] = {'precision': pr.tolist(),
                                       'recall': re.tolist(),
                                       'f-score': fs.tolist(),
                                       'labels': response_labels,
                                       'support': sup.tolist()}
    cf_matrix = pandas.DataFrame(confusion_matrix(expect, predict),
                                 index=response_labels,
                                 columns=response_labels)
    cf_matrix = cf_matrix.rename_axis('actual / predicted', axis=1)
    result["confusion_matrix"] = cf_matrix.to_dict(orient='list')
    result["au_roc"] = roc_auc_score(expect, predict)
    result["feature_params"] = feature_params
    outpath = os.path.join(output_path, "performance.json.gz")
    dump_json(outpath, result)


if __name__ == "__main__":
    main()
