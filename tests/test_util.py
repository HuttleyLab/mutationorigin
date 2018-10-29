from cogent3.util.unit_test import TestCase, main
from mutation_origin.util import get_enu_germline_sizes

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.3"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


class TestUtilityFuncs(TestCase):

    def test_get_sample_sizes(self):
        """check numerical transformation of data"""
        got = get_enu_germline_sizes(10, 1)
        self.assertEqual(got, (5, 5))
        got = get_enu_germline_sizes(11, 10)
        self.assertEqual(got, (10, 1))
        got = get_enu_germline_sizes(101, 100)
        self.assertEqual(got, (100, 1))


if __name__ == '__main__':
    main()
