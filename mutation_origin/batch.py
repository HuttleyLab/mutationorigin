"""command line interface for batched runs mutation_origin"""
from warnings import filterwarnings
from pprint import pprint
import os
os.environ['DONT_USE_MPI'] = "1"
filterwarnings("ignore", ".*Not using MPI.*")
import click
import pandas
from tqdm import tqdm
from cogent3 import LoadTable
from cogent3.util import parallel
from mutation_origin.cli import (sample_data as mutori_sample,
                                 lr_train as mutori_lr_train,
                                 nb_train as mutori_nb_train,
                                 ocs_train as mutori_ocs_train,
                                 predict as mutori_predict,
                                 performance as mutori_performance)
from mutation_origin.opt import (_seed, _feature_dim, _enu_path,
                                 _germline_path, _output_path, _flank_size,
                                 _train_size, _test_size, _enu_ratio,
                                 _numreps, _label_col, _proximal, _usegc,
                                 _training_path, _c_values, _penalty_options,
                                 _n_jobs, _classifier_paths, _data_path,
                                 _predictions_path, _alpha_options,
                                 _overwrite, _size_range, _model_range,
                                 _test_data_paths, _max_flank, _verbose)
from mutation_origin.util import (dirname_from_features, flank_dim_combinations,
                                  exec_command, FILENAME_PATTERNS,
                                  sample_size_from_path,
                                  data_rep_from_path,
                                  feature_set_from_path, load_json,
                                  summary_stat_table, model_name_from_features)
from scitrack import CachingLogger

LOGGER = CachingLogger()


@click.group()
def main():
    """mutori_batch -- batch execution of mutori subcommands"""
    pass


@main.command()
@_seed
@_enu_path
@_germline_path
@_output_path
@_enu_ratio
@_numreps
@_overwrite
@_size_range
@_n_jobs
@click.pass_context
def sample_data(ctx, enu_path, germline_path, output_path, seed,
                enu_ratio, numreps, overwrite, size_range, n_jobs):
    """batch creation training/testing sample data"""
    args = locals()
    args.pop('ctx')
    args.pop("n_jobs")
    args.pop("size_range")
    sizes = list(map(lambda x: int(x), size_range.split(",")))

    arg_sets = []
    for size in sizes:
        arg_group = args.copy()
        arg_group['train_size'] = size * 1000
        arg_group['output_path'] = os.path.join(output_path, f"{size}k")
        arg_sets.append(arg_group)

    if n_jobs > 1:
        parallel.use_multiprocessing(n_jobs)

    total = len(arg_sets)
    gen = parallel.imap(lambda args: ctx.invoke(mutori_sample,
                                                **args), arg_sets)
    for r in tqdm(gen, total=total):
        pass


def MakeDims(min_val=1, max_val=None):
    """factory function generating dimension ranges for provided flank_size"""
    def make_dim(flank_size):
        return [2 * flank_size]

    if min_val is None:
        return make_dim

    if max_val and max_val < min_val:
        raise ValueError

    start = min_val

    def make_dim(flank_size):
        stop = 2 * \
            flank_size if max_val is None else min(2 * flank_size, max_val)
        dims = list(range(start, stop + 1))
        return dims

    return make_dim


def get_train_kwarg_sets(training_path, output_path, max_flank,
                         model_range, usegc, proximal, args):
    """standadrised generation of kwargs for train algorithms"""
    get_dims = {'upto1': MakeDims(1, 1),
                'upto2': MakeDims(1, 2),
                'upto3': MakeDims(1, 3),
                'FS': MakeDims(None, None)}[model_range]
    start_flank = {'FS': 2}.get(model_range, 0)
    parameterisations = flank_dim_combinations(max_flank=max_flank,
                                               start_flank=start_flank,
                                               get_dims=get_dims)

    # find all the training data
    train_pattern = FILENAME_PATTERNS["sample_data"]["train"]
    cmnd = f'find {training_path} -name "{train_pattern}"'
    train_paths = exec_command(cmnd)
    train_paths = train_paths.splitlines()

    # we want to process smallest to largest samples
    train_paths.sort(key=sample_size_from_path)

    other_features = dict(usegc=usegc, proximal=proximal)
    arg_sets = []
    for train_path in train_paths:
        data_size = sample_size_from_path(train_path) // 1000
        data_size = f"{data_size}k"
        for params in parameterisations:
            params = params.copy()
            params.update(other_features)
            params.update(args)
            dim = params.get("feature_dim")
            flank_size = params["flank_size"]
            if dim is None or dim < 2 or flank_size < 2:
                # prox only sensible with dim >= 2, flank_size > 1
                params["proximal"] = False

            params['training_path'] = train_path
            params['output_path'] = os.path.join(output_path, data_size,
                                                 dirname_from_features(params))
            arg_sets.append(params)

    return arg_sets


@main.command()
@click.option('-tp', '--training_path',
              type=click.Path(exists=True),
              required=True,
              help='Input file containing training data.')
@_output_path
@_label_col
@_seed
@_max_flank
@_model_range
@_proximal
@_usegc
@_c_values
@_penalty_options
@_n_jobs
@_overwrite
@click.pass_context
def lr_train(ctx, training_path, output_path, label_col, seed,
             max_flank, model_range, proximal,
             usegc, c_values, penalty_options, n_jobs, overwrite):
    """batch logistic regression training"""
    args = locals()
    args.pop('ctx')
    args.pop("n_jobs")
    args.pop("max_flank")
    args.pop("model_range")

    arg_sets = get_train_kwarg_sets(training_path, output_path,
                                    max_flank, model_range,
                                    usegc, proximal, args)

    if n_jobs > 1:
        parallel.use_multiprocessing(n_jobs)

    total = len(arg_sets)
    gen = parallel.imap(lambda args: ctx.invoke(mutori_lr_train,
                                                **args), arg_sets)
    for r in tqdm(gen, total=total):
        pass


@main.command()
@_training_path
@_output_path
@_label_col
@_seed
@_max_flank
@_model_range
@_proximal
@_usegc
@_alpha_options
@_n_jobs
@_overwrite
@click.pass_context
def nb_train(ctx, training_path, output_path, label_col, seed,
             max_flank, model_range, proximal, usegc,
             alpha_options, n_jobs, overwrite):
    """batch naive bayes training"""
    args = locals()
    args.pop('ctx')
    args.pop("n_jobs")
    args.pop("max_flank")
    args.pop("model_range")

    arg_sets = get_train_kwarg_sets(training_path, output_path,
                                    max_flank, model_range,
                                    usegc, proximal, args)
    if n_jobs > 1:
        parallel.use_multiprocessing(n_jobs)

    total = len(arg_sets)
    gen = parallel.imap(lambda args: ctx.invoke(mutori_nb_train,
                                                **args), arg_sets)
    for r in tqdm(gen, total=total):
        pass


@main.command()
@_training_path
@_output_path
@_label_col
@_seed
@_max_flank
@_model_range
@_proximal
@_usegc
@_n_jobs
@_overwrite
@click.pass_context
def ocs_train(ctx, training_path, output_path, label_col, seed,
              max_flank, model_range, proximal, usegc, n_jobs, overwrite):
    """batch one class SVM training"""
    args = locals()
    args.pop('ctx')
    args.pop("n_jobs")
    args.pop("max_flank")
    args.pop("model_range")

    arg_sets = get_train_kwarg_sets(training_path, output_path, max_flank,
                                    model_range, usegc, proximal, args)
    if n_jobs > 1:
        parallel.use_multiprocessing(n_jobs)

    total = len(arg_sets)
    gen = parallel.imap(lambda args: ctx.invoke(mutori_ocs_train,
                                                **args), arg_sets)
    for r in tqdm(gen, total=total):
        pass


@main.command()
@_classifier_paths
@_test_data_paths
@_output_path
@_overwrite
@_n_jobs
@click.pass_context
def predict(ctx, classifier_paths, test_data_paths, output_path,
            overwrite, n_jobs):
    """batch testing of classifiers"""
    args = locals()
    args.pop('ctx')
    args.pop("n_jobs")
    args.pop("classifier_paths")
    args.pop("test_data_paths")

    class_pattern = FILENAME_PATTERNS["train"]
    classifier_fns = exec_command(
        f"find {classifier_paths} -name {class_pattern}")

    test_pattern = FILENAME_PATTERNS["sample_data"]["test"]
    data_fns = exec_command(f"find {test_data_paths} -name {test_pattern}")

    # create a dict from sample size, number
    data_fns = data_fns.splitlines()
    data_mapped = {}
    for path in data_fns:
        size = sample_size_from_path(path)
        size = f"{size // 1000}k"
        rep = data_rep_from_path("sample_data", path)
        data_mapped[(size, rep)] = path

    classifier_fns = classifier_fns.splitlines()
    paired = []
    for path in classifier_fns:
        size = sample_size_from_path(path)
        size = f"{size // 1000}k"
        rep = data_rep_from_path("train", path)
        featdir = feature_set_from_path(path)
        paired.append(dict(classifier_path=path,
                           data_path=data_mapped[(size, rep)],
                           size=size,
                           featdir=featdir))

    arg_sets = []
    for pair in paired:
        arg_group = args.copy()
        size = pair.pop('size')
        featdir = pair.pop('featdir')
        arg_group.update(pair)
        arg_group['output_path'] = os.path.join(output_path, size, featdir)
        arg_sets.append(arg_group)

    if n_jobs > 1:
        parallel.use_multiprocessing(n_jobs)

    total = len(arg_sets)
    gen = parallel.imap(lambda args: ctx.invoke(mutori_predict,
                                                **args), arg_sets)
    for r in tqdm(gen, total=total):
        pass


@main.command()
@_test_data_paths
@_predictions_path
@_output_path
@_label_col
@_overwrite
@_n_jobs
@_verbose
@click.pass_context
def performance(ctx, test_data_paths, predictions_path, output_path, label_col,
                overwrite, n_jobs, verbose):
    """batch classifier performance assessment"""
    args = locals()
    args.pop('ctx')
    args.pop("n_jobs")
    args.pop("test_data_paths")
    args.pop("predictions_path")
    args.pop("output_path")

    test_pattern = FILENAME_PATTERNS["sample_data"]["test"]
    predict_pattern = FILENAME_PATTERNS["predict"]
    test_fns = exec_command(f"find {test_data_paths} -name {test_pattern}")

    data_fns = test_fns.splitlines()
    data_mapped = {}
    for path in data_fns:
        size = sample_size_from_path(path)
        size = f"{size // 1000}k"
        rep = data_rep_from_path("sample_data", path)
        data_mapped[(size, rep)] = path

    predict_fns = exec_command(f'find {predictions_path} -name'
                               f' {predict_pattern}')
    predict_fns = predict_fns.splitlines()
    paired = []
    for path in predict_fns:
        size = sample_size_from_path(path)
        size = f"{size // 1000}k"
        rep = data_rep_from_path("train", path)
        featdir = feature_set_from_path(path)
        paired.append(dict(predictions_path=path,
                           data_path=data_mapped[(size, rep)],
                           size=size,
                           featdir=featdir))

    arg_sets = []
    for pair in paired:
        arg_group = args.copy()
        size = pair.pop('size')
        featdir = pair.pop('featdir')
        arg_group.update(pair)
        arg_group['output_path'] = os.path.join(output_path, size, featdir)
        arg_sets.append(arg_group)

    if n_jobs > 1:
        parallel.use_multiprocessing(n_jobs)

    total = len(arg_sets)
    gen = parallel.imap(lambda args: ctx.invoke(mutori_performance,
                                                **args), arg_sets)
    for r in tqdm(gen, total=total):
        pass


@main.command()
@click.option('-bp', '--base_path',
              type=click.Path(exists=True),
              help='Base directory containing all'
              ' files produced by performance.')
@_output_path
@_overwrite
def collate(base_path, output_path, overwrite):
    """collates all classifier performance stats and writes
    to a single tsv file"""
    LOGGER.log_args()
    outpath = os.path.join(output_path, "collated.tsv.gz")
    logfile_path = os.path.join(output_path, "collated.log")
    if os.path.exists(outpath) and not overwrite:
        click.secho(f"Skipping. {outpath} exists. "
                    "Use overwrite to force.",
                    fg='green')
        exit(0)

    stat_fns = exec_command(f'find {base_path} -name'
                            ' "*performance.json*"')
    stat_fns = stat_fns.splitlines()
    if not stat_fns:
        msg = f'No files matching "*performance.json*" in {base_path}'
        click.secho(msg, fg='red')
        return

    LOGGER.log_file_path = logfile_path

    records = []
    keys = set()
    for fn in tqdm(stat_fns):
        LOGGER.input_file(fn)
        data = load_json(fn)
        row = {"stat_path": fn, "classifier_path": data["classifier_path"],
               "auc": data["auc"], "algorithm": data["classifier_label"]}
        row.update(data["feature_params"])
        keys.update(row.keys())
        records.append(row)

    columns = sorted(keys)
    rows = list(map(lambda r: [r.get(c, None) for c in columns], records))
    table = LoadTable(header=columns, rows=rows)
    table = table.sorted(reverse="auc")
    table = table.with_new_column("name",
                                  lambda x: model_name_from_features(*x),
                                  columns=["flank_size", "feature_dim",
                                           "usegc", "proximal"])
    table = table.with_new_column("size", sample_size_from_path,
                                  columns="classifier_path")
    table.write(outpath)
    LOGGER.output_file(outpath)

    # make summary statistics via grouping by factors
    factors = ["algorithm", "name", "flank_size", "feature_dim",
               "proximal", "usegc", "size"]
    summary = summary_stat_table(table, factors=factors)
    outpath = os.path.join(output_path, "summary_statistics.tsv.gz")
    summary.write(outpath)
    LOGGER.output_file(outpath)


if __name__ == '__main__':
    main()
