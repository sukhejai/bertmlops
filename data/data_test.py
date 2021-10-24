# Test integrity of the input data

import os
import numpy as np
import pandas as pd

# get absolute path of csv files from data folder


def get_absPath(filename):
    """Returns the path of the notebooks folder"""
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(
                __file__), os.path.pardir, "data", filename
        )
    )
    return path


# number of features
expected_columns = 4


def test_check_schema():
    datafile = get_absPath("bert_classification_input.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    header = dataset[dataset.columns[:-1]]
    actual_columns = header.shape[1]
    # check header has expected number of columns
    assert actual_columns == expected_columns


def test_check_bad_schema():
    datafile = get_absPath("bert_classification_bad_schema.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    header = dataset[dataset.columns[:-1]]
    actual_columns = header.shape[1]
    # check header has expected number of columns
    assert actual_columns != expected_columns


def test_check_missing_values():
    datafile = get_absPath("bert_classification_missing_values.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    n_nan = np.sum(np.isnan(dataset.values))
    assert n_nan > 0
