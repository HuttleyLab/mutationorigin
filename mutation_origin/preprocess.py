import pandas
import numpy
from mutation_origin.feature import (feature_selector, seq_feature_labels_upto,
                                     get_mutation_direction_labels)
from mutation_origin.encoder import transformed, onehot, transform_response
from mutation_origin.util import valid_response_values

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.1"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


def data_to_numeric(path, label_col=None, flank_size=None,
                    feature_dim=None, proximal=False, usegc=False,
                    one_class=None):
    """returns one-hot encoded data and vector of dimensions"""
    data = pandas.read_csv(path, sep='\t')

    if label_col and not valid_response_values(data[label_col]):
        raise ValueError("response not denoted by 'e'/'g'")

    if label_col and one_class:
        data = data[data[label_col] == one_class]

    identifiers = data['varid'].tolist()
    resp, feat = feature_selector(data, label_col,
                                  flank_size,
                                  feature_dim, proximal)

    if resp is not None:
        resp = transform_response(resp)

    # get the seq feature labels
    labels = seq_feature_labels_upto(feature_dim)
    # and the mutation direction labels
    le = get_mutation_direction_labels()

    # numerically encode the sequence features
    feat, n_dims, names = transformed(feat, labels,
                                      direction_transform=le)
    # now one-hot encode
    feat = onehot(feat, n_dims)

    # if usegc, extract that column and insert into feature
    if usegc:
        gc_col = "GC" if "GC" in data else "gc"

        if gc_col not in data:
            raise ValueError("no column label GC")

        gc = data[gc_col].tolist()
        feat = numpy.insert(feat, 0, gc, axis=1)

    return identifiers, resp, feat, n_dims, names
