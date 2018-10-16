import numpy as np
from sklearn.preprocessing import LabelEncoder
from cogent3.util.unit_test import TestCase, main
from mutation_origin.encoder import (transformed, onehot,
                             transform_response,
                             inverse_transform_response)
from mutation_origin.feature import (get_mutation_direction_labels,
                             seq_feature_labels_upto,
                             feature_indices_upto,
                             seq_2_features)

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.1"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


class TestEncoder(TestCase):

    def test_transformed(self):
        """check numerical transformation of data"""
        seq_features = [['A', 'C'],
                        ['G', 'T']]
        labels = seq_feature_labels_upto(2)
        result, n_values, names = transformed(seq_features, labels)
        self.assertEqual(result.tolist(), [[0, 1], [2, 3]])
        self.assertEqual(n_values, [4, 4])
        seq_features = [['A', 'C', 'TT'],
                        ['G', 'T', 'AA']]
        result, n_values, names = transformed(seq_features, labels)
        self.assertEqual(result.tolist(), [[0, 1, 15], [2, 3, 0]])
        self.assertEqual(n_values, [4, 4, 16])

        # check when there's a mutation direction
        le = get_mutation_direction_labels()
        seq_features = [['AtoC', 'A', 'C'],
                        ['TtoG', 'G', 'T']]
        result, n_values, names = transformed(
            seq_features, labels, direction_transform=le)
        self.assertEqual(result.tolist(), [[0, 0, 1], [11, 2, 3]])
        self.assertEqual(n_values, [12, 4, 4])

    def test_onehot(self):
        """exercise onehot encoding of data"""
        data_matrix = [['AtoC', 'CCCC'],
                       ['TtoC', 'CACC'],
                       ['GtoA', 'TGTG'],
                       ['CtoT', 'AACA'],
                       ['AtoG', 'AGCA']]
        indices = feature_indices_upto(2, 1)
        features = []
        for row in data_matrix:
            features.append([row[0]] + seq_2_features(row[1], indices))

        labels = seq_feature_labels_upto(1)
        md = get_mutation_direction_labels()
        numeric, n_values, names = transformed(features, labels,
                                               direction_transform=md)
        ohot = onehot(numeric, n_values=n_values)

    def test_transform_response(self):
        """correctly transform response values and invert"""
        orig = ['e', 'g', 'g', 'e']
        got = transform_response(orig)
        self.assertEqual(got, [-1, 1, 1, -1])
        orig = [1, 1, 1, -1, 1]
        got = inverse_transform_response(orig)
        self.assertEqual(got, ['g', 'g', 'g', 'e', 'g'])


if __name__ == '__main__':
    main()
