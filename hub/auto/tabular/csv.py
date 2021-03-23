import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import hub

from hub.auto import util
from hub.auto.infer import state


@state.directory_parser(priority=1)
def data_from_csv(path, scheduler, workers):

    # check if path's contents are all csv files
    if not util.files_are_of_extension(path, util.CSV_EXTS):
        return None

    df = pd.DataFrame()
    files = util.get_children(path)

    for i in files:
        df_csv = pd.read_csv(i)
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
    # print(f"Your schema is: {schema}")

    @hub.transform(schema=schema, scheduler=scheduler, workers=workers)
    def csv_data(index, df):
        dictionary_cols = {}
        for column in df.columns:
            dictionary_cols[column] = df[column].iloc[index]
        return dictionary_cols

    return csv_data(range(len(df)), df=df)
