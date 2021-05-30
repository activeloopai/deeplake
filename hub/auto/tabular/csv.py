import os

import hub
import numpy as np
from hub.auto import util
from hub.auto.infer import state
from hub.exceptions import ModuleNotInstalledException
from tqdm import tqdm


@state.directory_parser(priority=1)
def data_from_csv(path, scheduler, workers, **kwargs):
    sep = kwargs["sep"] if "sep" in kwargs else ","
    try:
        import pandas as pd
    except ModuleNotFoundError:
        raise ModuleNotInstalledException("Please install Pandas for CSV parsing.")

    # check if path's contents are all csv files
    if not util.files_are_of_extension(path, util.CSV_EXTS):
        return None

    df = pd.DataFrame()
    files = util.get_children(path)

    for i in files:
        try:
            df_csv = pd.read_csv(i, sep)
        except pd.errors.ParserError:
            raise pd.errors.ParserError(
                f"Another seperator needs to be used to parse this CSV. The current seperator is {sep}"
            )
        df_csv["Filename"] = os.path.basename(i)
        df = pd.concat([df, df_csv])

    schema = {str(i): df[i].dtype for i in df.columns}
    for keys in schema.keys():
        if schema[keys] == np.dtype("O"):
            # Assigning max_shape as the length of the longest string in the column.
            schema[keys] = hub.schema.Text(
                shape=(None,), max_shape=(int(df[keys].str.len().max()),)
            )
        # the below code is to check whether the column is a ClassLabel or not
        # elif schema[keys] == np.dtype("int64"):
        #     if len(np.unique(df[keys])) <= 10:
        #         schema[keys] = hub.schema.ClassLabel(
        #             num_classes=len(np.unique(df[keys]))
        #         )
        #     else:
        #         schema[keys] = hub.schema.Primitive(dtype=schema[keys])
        else:
            schema[keys] = hub.schema.Primitive(dtype=schema[keys])

    @hub.transform(schema=schema, scheduler=scheduler, workers=workers)
    def upload_data(index, df):
        dictionary_cols = {}
        for column in df.columns:
            dictionary_cols[column] = df[column].iloc[index]
        return dictionary_cols

    return upload_data(range(len(df)), df=df)
