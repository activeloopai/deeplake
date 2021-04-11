import os

import hub
import numpy as np
from hub.auto import util
from hub.auto.infer import state
from tqdm import tqdm

USE_TQDM = True


@state.directory_parser(priority=2)
def data_from_audio(path, scheduler, workers):

    try:
        import librosa
    except ModuleNotFoundError:
        raise ModuleNotInstalledException("librosa")

    try:
        import pandas as pd
    except ModuleNotFoundError:
        raise ModuleNotInstalledException("pandas")

    if not util.files_are_of_extension(path, AUDIO_EXTS):
        return None

    max_shape = (0,)

    df = pd.DataFrame()
    files = util.get_children(path)

    for audio_file in files:
        audio, sr = librosa.load(audio_file)
        df = df.append({"Audio": audio, "Sampling Rate": sr}, ignore_index=True)
        shape = audio.shape
        max_shape = np.maximum(max_shape, shape)

    max_shape = tuple([int(x) for x in max_shape])

    schema = {
        "audio": hub.schema.Audio(shape=(None,), dtype="uint8", max_shape=max_shape),
        "sampling_rate": hub.schema.Primitive(dtype=int),
    }

    # Create transform for putting data into hub format
    @hub.transform(schema=schema, scheduler=scheduler, workers=workers)
    def upload_data(index, df):
        audio_dictionary = {}
        for column in df.columns:
            audio_dictionary[column] = df[column].iloc[index]
        return audio_dictionary

    return upload_data(range(len(df)), df=df)
