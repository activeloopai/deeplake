import json
import os
import shutil
import zipfile
from pathlib import PosixPath

import hub
import numpy as np
import pytest
from hub.auto.tests.util import get_dataset_store
from hub.utils import pandas_loaded
from hub.auto.util import get_children


def assert_conversion(tag):
    import pandas as pd

    dataset_store = get_dataset_store(tag)
    hub_dir = dataset_store / "hub"

    # delete hub dataset so conversion test can be done
    if hub_dir.is_dir():
        print("hub_dir was found (%s), deleting..." % hub_dir)
        shutil.rmtree(hub_dir)

    df = pd.DataFrame()
    files = get_children(dataset_store)
    for i in files:
        df_csv = pd.read_csv(i)
        df_csv["Filename"] = os.path.basename(i)
        df = pd.concat([df, df_csv])

    try:
        ds = hub.Dataset.from_path(str(dataset_store))
    except Exception:
        assert False

    print("dataset obj:", ds)
    assert ds is not None

    assert hub_dir.is_dir(), hub_dir

    # df = Pandas dataframe, ds = Dataset obtained from hub.auto
    if df is not None:
        assert ds.shape == (df.shape[0],)

    # Checking if the column names are the same
    keys_csv_parser = [i[1:] for i in ds.keys]
    keys_df = list(df.columns)
    assert keys_csv_parser == keys_df

    # Checking if all elements are parsed correctly
    for i in keys_df:
        column = []
        if df[i].dtype == np.dtype("O"):
            for j in range(df.shape[0]):
                column.append(ds[i, j].compute())
        else:
            column = ds[i].compute()
        assert list(column) == list(df[i])

    # Checking if the datatypes of the columns match
    for i in keys_csv_parser:
        if df[i].dtype == np.dtype("O"):
            assert type(ds[0, i].compute()) == str
        else:
            assert str(ds[0, i].compute().dtype) == str(df[i].dtype)

    # Checking if all the filenames are parsed correctly
    list_names = []
    for i in range(len(ds)):
        if ds["Filename", i].compute() in list_names:
            continue
        list_names.append(ds["Filename", i].compute())
    assert list(df["Filename"].unique()) == list_names


@pytest.mark.skipif(not pandas_loaded(), reason="requires pandas to be loaded")
def test_class_sample_single_csv():
    tag = "tabular/single_csv"
    assert_conversion(tag)


@pytest.mark.skipif(not pandas_loaded(), reason="requires pandas to be loaded")
def test_class_sample_multiple_csv():
    tag = "tabular/multiple_csv"
    assert_conversion(tag)
