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
