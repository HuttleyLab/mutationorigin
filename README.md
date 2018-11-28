# `mutation_origin`: python library for classifying mutation origins

This repository contains scripts used to train and deploy the classification analyses reported in *Classifying ENU induced mutations from spontaneous germline mutations in mouse with machine learning techniques* by Zhu, Ong and Huttley.

## Initial Setup

Inside a `conda` environment, run pip on the downloaded zip file.

```$ pip install mutori.zip```

## The `mutori` and `mutori_batch` commands

Installation creates `mutori` and `mutori_batch` command line scripts. Command line help for `mutori` shows

```
$ mutori
Usage: mutori [OPTIONS] COMMAND [ARGS]...

  mutori -- for building and applying classifiers of mutation origin

Options:
  --help  Show this message and exit.

Commands:
  lr_train       logistic regression training, validation,...
  nb_train       Naive Bayes training, validation, dumps...
  ocs_train      one-class svm training for outlier detection
  performance    produce measures of classifier performance
  predict        predict labels for data
  sample_data    creates train/test sample data
  xgboost_train  Naive Bayes training, validation, dumps...
```

 Command line help for `mutori_batch` shows

```
$ mutori_batch 
Usage: mutori_batch [OPTIONS] COMMAND [ARGS]...

  mutori_batch -- batch execution of mutori subcommands

Options:
  --help  Show this message and exit.

Commands:
  collate        collates all classifier performance stats and...
  lr_train       batch logistic regression training
  nb_train       batch naive bayes training
  ocs_train      batch one class SVM training
  performance    batch classifier performance assessment
  predict        batch testing of classifiers
  sample_data    batch creation training/testing sample data
  xgboost_train  batch xgboost training
```

### Input and output data formats

#### Input sequence data

Must be in a tab delimited form, with a header line. The file will be read by `pandas.read_csv`. Required columns are: `varid`, variant identifiers; `flank5` and `flank3` are the 5' and 3' flanking sequences respectively; `direction`, mutation direction with values of form `XtoY` (X and Y are nucleotides).

For training, the file must also contain a `response` column containing either `e`/`g` (for ENU and spontaneous Germline respectively)

If the GC% is to be examined, a `GC` column is also required.

The column order does not matter.

#### Classifiers

These are saved in python's `pickle` format. Also saved are attributes defining the feature set against which the classifier was trained.

#### Performance measures

Stored in `json` format.

#### Collated output

Done via the `mutori_batch collate` command, produces tab separated files of key performance metrics and summary statistics of each of those.

## License

The BSD 3-clause license is included in this repo as well, refer to `license.txt`
