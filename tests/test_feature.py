from cogent3.util.unit_test import TestCase, main
from mutation_origin.feature import (isproximal, get_feature_indices,
                             feature_indices_upto, seq_2_features,
                             seq_feature_labels_upto)

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.3"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


class TestEncoder(TestCase):

    def test_isproximal(self):
        """given a dimension value, return columns in a feature matrix with
         defined dimensional neighborhood"""
        self.assertTrue(isproximal([1]))
        self.assertTrue(isproximal([0, 1]))
        self.assertTrue(isproximal([1, 0]))
        self.assertTrue(isproximal([0, 1, 2]))
        self.assertTrue(isproximal([2, 3, 4]))

        self.assertFalse(isproximal([0, 2]))
        self.assertFalse(isproximal([0, 3, 4]))
        self.assertFalse(isproximal([0, 1, 3]))
        self.assertFalse(isproximal([1, 2, 4]))

    def test_get_feature_indices(self):
        """selecting feature indices"""
        # case of single positions
        got = get_feature_indices(2, 1)
        self.assertEqual(got, [(0,), (1,), (2,), (3,)])
        got = get_feature_indices(2, 2)
        self.assertEqual(got, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        got = get_feature_indices(2, 2, proximal=True)
        self.assertEqual(got, [(0, 1), (1, 2), (2, 3)])

    def test_feature_indices_upto(self):
        """correctly produces all feature indices"""
        got = feature_indices_upto(4, 1, proximal=False)
        self.assertEqual(got, [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)])
        got = feature_indices_upto(2, 2, proximal=False)
        self.assertEqual(got, [(0,), (1,), (2,), (3,),
                               (0, 1), (0, 2), (0, 3),
                               (1, 2), (1, 3),
                               (2, 3)])
        got = feature_indices_upto(2, 2, proximal=True)
        self.assertEqual(got, [(0,), (1,), (2,), (3,),
                               (0, 1), (1, 2), (2, 3)])

        got = feature_indices_upto(2, 3, proximal=False)
        self.assertEqual(got, [(0,), (1,), (2,), (3,),
                               (0, 1), (0, 2), (0, 3),
                               (1, 2), (1, 3),
                               (2, 3),
                               (0, 1, 2), (0, 1, 3),
                               (0, 2, 3), (1, 2, 3)])

    def test_seq_2_features(self):
        """convert a sequence to string features"""
        seq = "CAGA"
        indices = feature_indices_upto(2, 1, proximal=False)
        got = seq_2_features(seq, indices)
        self.assertEqual(got, ['C', 'A', 'G', 'A'])
        indices = feature_indices_upto(2, 2, proximal=False)
        got = seq_2_features(seq, indices)
        self.assertEqual(got, ['C', 'A', 'G', 'A',
                               'CA', 'CG', 'CA',
                               'AG', 'AA', 'GA'])
        indices = feature_indices_upto(2, 2, proximal=True)
        got = seq_2_features(seq, indices)
        self.assertEqual(got, ['C', 'A', 'G', 'A',
                               'CA', 'AG', 'GA'])

    def test_seq_feature_labels_upto(self):
        """construction of sequence feature labels"""
        # the possible labels for a given dimension
        # is just the k-mers for that dimension
        for dim in range(1, 5):
            got = seq_feature_labels_upto(dim)
            for i in range(1, dim + 1):
                self.assertEqual(4**i, len(got[i].classes_))


if __name__ == '__main__':
    main()
