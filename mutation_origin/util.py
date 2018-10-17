import json
import pandas
from cogent3.util.misc import open_

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


def load_predictions(infile_path):
    """returns dataframe, params from  json format prediction data"""
    with open_(infile_path) as infile:
        data = json.load(infile)

    params = data["feature_params"]
    df = pandas.DataFrame(data["predictions"])
    return df, params
