import os
import json
import pandas
from cogent3.util.misc import open_, get_format_suffixes

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.1"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


MUTATION_DIRECTIONS = ('AtoC', 'AtoG', 'AtoT', 'CtoA', 'CtoG', 'CtoT',
                       'GtoA', 'GtoC', 'GtoT', 'TtoA', 'TtoC', 'TtoG')
BASES = tuple("ACGT")


def valid_response_values(data):
    vals = set(data)
    return vals <= {'e', 'g'}


def dump_json(path, data):
    """dumps data in json format"""
    with open_(path, mode="wt") as outfile:
        json.dump(data, outfile)


def load_json(path):
    """loads raw data object from json file"""
    with open_(path) as infile:
        data = json.load(infile)
    return data


def load_predictions(path):
    """returns dataframe, params from  json format prediction data"""
    data = load_json(path)

    params = data["feature_params"]
    df = pandas.DataFrame(data["predictions"])
    cpath = data.get("classifier_path", None)
    label = data.get("classifier_label", None)
    return df, params, cpath, label


def get_basename(path):
    """returns a file basename without the suffixes"""
    bn = os.path.basename(path)
    suffix, cmp_suffix = get_format_suffixes(bn)
    rindex = bn.rfind(f".{suffix}")
    return bn[:rindex]


def get_classifier_label(classifier):
    """returns string label of classifier"""
    name = classifier.__class__.__name__.lower()
    if "logistic" in name:
        label = 'lr'
    elif 'nb' in name:
        label = 'nb'
    elif 'svm' in name:
        label = 'ocs'
    else:
        raise ValueError("Unknown classifier type {name}")
    return label
