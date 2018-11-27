import pandas
from sklearn.metrics import (confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support,
                             precision_recall_curve)
from mutation_origin.encoder import (inverse_transform_response,
                                     transform_response)

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.3"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


def measure_performance(orig, predicted, label_col):
    """returns dict of classifier performance measures

    Arguments:
    - orig: DataFrame of original data
    - predicted: DataFrame of predictions
    - label_col: column containing the labels"""
    orig = orig[['varid', label_col]]
    orig.set_index('varid', inplace=True)
    predicted.set_index('varid', inplace=True)
    new = orig.join(predicted)

    response_labels = inverse_transform_response([-1, 1])
    expect = transform_response(new[label_col])
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
    result["auc"] = roc_auc_score(expect, new['scores'])
    precision, recall, thresholds = precision_recall_curve(
        expect, new['scores'])
    result["prcurve"] = dict(precision=precision.tolist(),
                             recall=recall.tolist(),
                             thresholds=thresholds.tolist())
    return result
