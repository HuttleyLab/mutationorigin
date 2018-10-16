import click

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.1"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


def _make_number(ctx, param, value):
    """converts a number to float/int"""
    if value:
        try:
            value = int(value)
        except ValueError:
            value = float(value)
    return value


def _make_num_series(ctx, param, value):
    value = [float(c) for c in value.split(',')]
    return value


_seed = click.option('-S', '--seed', type=int, default=None,
                     help='Seed for random number generator.'
                     ' Defaults to system time.')
_feature_dim = click.option('-d', '--feature_dim', type=click.IntRange(0, 21),
                            default=1,
                            help='Max dependent interaction order/dimension '
                            'considered when constructing position features.')
_enu_path = click.option('-e', '--enu_path', required=True,
                         type=click.Path(exists=True),
                         help='file path for tab delimited ENU mutation data.')
_germline_path = click.option('-g', '--germline_path', required=True,
                              type=click.Path(exists=True),
                              help='file path for tab delimited germline'
                              ' mutation data.')
_ouput_path = click.option('-o', '--ouput_path', required=True,
                           help='Path to write output.')
_flank_size = click.option('-f', '--flank_size', type=int, required=True,
                           help='flank size considered when query the'
                           ' data file')
_proximal = click.option('-p', '--proximal', is_flag=True,
                         help='only proximal positions used')
_usegc = click.option('--usegc', is_flag=True,
                      help='use GC% as a feature')
_train_size = click.option('--train_size', required=True, type=int,
                           help='the total number of train samples.')
_test_size = click.option('--test_size', required=True, type=int,
                          callback=_make_number,
                          help='the total number of test samples.')
_enu_ratio = click.option('-r', '--enu_ratio',
                          type=click.Choice(['1', '10', '100']), default='1',
                          callback=_make_number,
                          help='Ratio of ENU to germline in testing data.')
_numreps = click.option('-n', '--numreps', required=True, type=int,
                        help='Number of times to run the splitting'
                        ' process for.')
_label_col = click.option('-l', '--label_col', default='response',
                          help='Table column corresponding to response label.')
_training_path = click.option('-T', '--training_path',
                              type=click.Path(exists=True),
                              help='Input file containing training data.')
_data_path = click.option('-D', '--data_path',
                          type=click.Path(exists=True),
                          help='Input file containing training data.')
_c_values = click.option('-C', '--c_values',
                         default='0.1,1,10,100',
                         callback=_make_num_series,
                         help='C values choosed for '
                         'model, e.g. "0.1,1,10,100"')
_penalty_options = click.option('-P', '--penalty_options',
                                default='l1',
                                help="penalty parameter choosed for model, "
                                "e.g. 'l1','l2', or 'l1,l2'")
_alpha_options = click.option('-a', '--alpha_options',
                              default="0.01,0.1,1,2,3",
                              callback=_make_num_series,
                              help='Alpha values for model')
_n_jobs = click.option('--n_jobs', type=int, default=1,
                       help="num parallel jobs via joblib")
_classifier_path = click.option('-cp', '--classifier_path',
                                type=click.Path(exists=True),
                                help="path to pickle format classifier")
_predictions_path = click.option('-pp', '--predictions_path',
                                 type=click.Path(exists=True),
                                 help="path to output from predict")
