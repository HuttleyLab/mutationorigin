"""define features"""
from itertools import combinations, product
from scipy.special import binom
from sklearn.preprocessing import LabelEncoder
from mutation_origin.util import BASES, MUTATION_DIRECTIONS

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.2"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


def isproximal(coords):
    """returns True if all components are within 1"""
    result = True
    if len(coords) != 1:
        for i in range(1, len(coords)):
            diff = abs(coords[i] - coords[i - 1])
            if diff > 1:
                result = False
                break
    return result


def get_feature_indices(flank_size, feature_dim, proximal=False):
    """returns indices for features with a given flank size and feature dim

    Arguments:
        - flank_size: matched arouhnd a point mutation location
        - feature_dim: dimension of the features, e.g. 2 is all pairwise
          bases
        - proximal: whether only features consisting of proximal indices
          are allowed
    """
    def ignore_proximal(val):
        return True

    indices = list(range(2 * flank_size))
    func = isproximal if proximal else ignore_proximal
    coords = [idx for idx in combinations(indices, r=feature_dim)
              if func(idx)]

    return coords


def get_kmers(k):
    """returns all DNA k-mers

    Used for label encoding"""
    kmers = [''.join(b) for b in product(BASES, repeat=k)]
    return kmers


def feature_indices_upto(flank_size, feature_dim, proximal=False):
    """returns all feature indices up to feature_dim"""
    indices = []
    for dim in range(1, feature_dim + 1):
        indices.extend(get_feature_indices(flank_size,
                                           dim,
                                           proximal=proximal))
    return indices


def seq_feature_labels_upto(feature_dim):
    """return LabelEncoder instances for each sequence feature dimension"""
    labels = {}
    for dim in range(1, feature_dim + 1):
        le = LabelEncoder()
        le.fit(get_kmers(dim))
        labels[dim] = le

    return labels


def get_mutation_direction_labels():
    """returns the LabelEncoder instance for mutation direction"""
    le = LabelEncoder()
    le.fit(MUTATION_DIRECTIONS)
    return le


def sort_col_by_dim(features, flank_window_size, feature_dim):
    """given a feature_dim value, return columns in a feature matrix
    with defined feature_dim neighborhood"""
    def get_binomial_cofficient(x):
        """returns the binomial coefficient as an int"""
        return int(binom(flank_window_size, x))

    start_idx = sum(map(get_binomial_cofficient, range(feature_dim)))
    end_idx = sum(map(get_binomial_cofficient, range(1, feature_dim + 1))) + 1
    return features[:, start_idx: end_idx]


def seq_2_features(seq, indices):
    """converts a single sequence to a list of features
    based on indices
    """
    result = []
    for idxs in indices:
        feature = ''.join([seq[i] for i in idxs])
        result.append(feature)
    return result


def feature_selector(data, label_col, flank_size, feature_dim,
                     proximal):
    """returns mutation direction + seq features and the response (label) array

    Arguments:
        - label_col: column with response label
        """
    response = None if label_col is None else data[label_col]
    direction = data['direction'].tolist()
    indices = feature_indices_upto(flank_size, feature_dim, proximal)
    features = []
    flank5 = data['flank5'].tolist()
    flank3 = data['flank3'].tolist()
    for i in range(len(flank5)):
        if flank_size:
            seq = flank5[i][-flank_size:] + flank3[i][:flank_size]
            feat = seq_2_features(seq, indices)
        else:
            feat = []
        features.append([direction[i]] + feat)

    # `features` consists of the mutation direction and conversion of the
    # sequence into component features
    return response, features
