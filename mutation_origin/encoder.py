import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.3"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


def transformed(features, labels, direction_transform=None):
    """transforms sequence featurs according to the label encoders

    Assumes mutation direction at column 0"""
    features = np.array(features)
    funcs = []
    n_values = []
    if direction_transform:
        funcs.append(direction_transform)
        start = 1
        n_values.append(len(direction_transform.classes_))
    else:
        start = 0

    # determine the order of label transformers
    for c in range(start, len(features[0])):
        dim = len(features[0][c])
        funcs.append(labels[dim])
        n_values.append(len(labels[dim].classes_))

    names = []
    features = features.T  # allows row based transformation
    numeric = []
    for i, column in enumerate(features):
        numeric.append(funcs[i].transform(column))
        names.extend([f"({i}, {j})" for j in funcs[i].classes_.tolist()])
    numeric = np.array(numeric)
    numeric = numeric.T
    return numeric, n_values, names


def onehot(data, n_values):
    """encode as one-hot encoding"""
    encoder = OneHotEncoder(categories=[range(v) for v in n_values], dtype=np.int8)
    result = encoder.fit_transform(data).toarray()
    result[result == 0] = -1
    return result


def get_scaler(features):
    """converts a feature array into a continuous metric"""
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def transform_response(data):
    """converts e/g to -1/1"""
    response = [1 if v == 'g' else -1 for v in data]
    return response


def inverse_transform_response(data):
    """converts -1/1 to e/g"""
    response = ['g' if v == 1 else 'e' for v in data]
    return response
