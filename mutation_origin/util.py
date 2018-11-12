import re
import sys
from subprocess import Popen, PIPE
import os
import pickle
import json
import pandas
import numpy
from tqdm import tqdm
from cogent3.util.misc import open_, get_format_suffixes
from cogent3 import LoadTable


__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.3"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


FILENAME_PATTERNS = {"sample_data": dict(train="train-*.tsv.gz",
                                         test="test-*.tsv.gz"),
                     "train": "*-classifier-*.pkl",
                     "predict": "*-predicted-*.json.gz",
                     "performance": "*-performance.json.gz"}
MUTATION_DIRECTIONS = ('AtoC', 'AtoG', 'AtoT', 'CtoA', 'CtoG', 'CtoT',
                       'GtoA', 'GtoC', 'GtoT', 'TtoA', 'TtoC', 'TtoG')
BASES = tuple("ACGT")


def exec_command(cmnd, stdout=PIPE, stderr=PIPE):
    """executes shell command and returns stdout if completes exit code 0

    Parameters
    ----------

    cmnd : str
      shell command to be executed
    stdout, stderr : streams
      Default value (PIPE) intercepts process output, setting to None
      blocks this."""
    proc = Popen(cmnd, shell=True, stdout=stdout, stderr=stderr)
    out, err = proc.communicate()
    if proc.returncode != 0:
        msg = err
        sys.stderr.writelines("FAILED: %s\n%s" % (cmnd, msg))
        sys.exit(proc.returncode)
    if out is not None:
        r = out.decode('utf8')
    else:
        r = None
    return r


def get_enu_germline_sizes(total, enu_ratio):
    """returns the enu and germline sample sizes to satisfy enu_ratio"""
    unit = total / (enu_ratio + 1)
    enu = int(unit * enu_ratio)
    germline = total - enu
    return enu, germline


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


def load_classifier(path):
    """returns dict of pickled classifier and features info"""
    with open(path, 'rb') as clf:
        classifier = pickle.load(clf)
    try:
        feature_params = classifier["feature_params"]
        scaler = classifier.get('scaler', None)
        classifier = classifier["classifier"]
    except KeyError:
        raise ValueError("pickle formatted file does not "
                         "contain classifier")
    return classifier, feature_params, scaler



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


def dirname_from_features(features):
    """generates directory names from a feature set"""
    dirname = f"f{features['flank_size']}"
    if features.get('feature_dim'):
        dirname += f"d{features['feature_dim']}"
    if features.get('proximal'):
        dirname += 'p'
    if features.get('usegc'):
        dirname += 'GC'
    return dirname


def flank_dim_combinations(max_flank=4, start_flank=0, get_dims=None):
    """returns flank_size/dim combinations"""
    combinations = []
    for fz in range(start_flank, max_flank):
        if fz == 0:
            combinations.append(dict(flank_size=fz))
            continue

        if get_dims is None:
            dims = range(1, 2 * fz)
        else:
            dims = get_dims(fz)

        for dim in dims:
            combinations.append(dict(flank_size=fz, feature_dim=dim))

    return combinations


_size = re.compile(r"(?<=/)\d+(?=k/)")


def sample_size_from_path(path):
    """returns component of path ijndicating sample size"""
    try:
        size = int(_size.findall(path)[0]) * 1000
    except IndexError:
        size = None
    return size


_rep = re.compile(r"(?<=-)\d+(?=[-.])")


def data_rep_from_path(src, path):
    """returns component of path indicating sample size"""
    basename = os.path.basename(path)
    rep = _rep.findall(basename)[0]
    return rep


_feats = re.compile(r"f\d+[d\d]*p*[GC]*")


def feature_set_from_path(path):
    features = _feats.findall(path)[0]
    return features


def model_name_from_features(flank, dim, usegc, proximal):
    """returns the model name from a feature set"""
    if flank == 0:
        model = ["MD"]
    elif flank and dim == 1:
        model = ["MD", "I"]
    elif flank and dim == 2 * flank:
        assert not proximal
        model = ["FS"]
    elif dim > 1:
        prox = "p" if proximal else ""
        model = ["MD", "I"] + [f"{i}D{prox}" for i in range(2, dim + 1)]
    else:
        raise ValueError("Unexpected model", flank, flank)
    if usegc:
        model.append("GC")

    name = "+".join(model)
    return name


def summary_stat_table(table, factors):
    """returns summary statistics for classifier, feature set combination"""
    distinct = table.distinct_values(factors)
    rows = []
    for comb in tqdm(distinct):
        subtable = table.filtered(lambda x: tuple(x) == tuple(comb),
                                  columns=factors)
        aurocs = numpy.array(subtable.tolist('auc'))
        rows.append(list(comb) + [aurocs.mean(), aurocs.std(ddof=1)])

    table = LoadTable(header=list(factors) +
                      ["mean_auc", "std_auc"], rows=rows)
    table = table.sorted(reverse="mean_auc")
    return table


def iter_indices(total, block_size):
    """yields a block_size numpy array of indices up to total"""
    for start in range(0, total, block_size):
        end = min(total, start + block_size)
        indices = range(start, end)
        yield list(indices)
