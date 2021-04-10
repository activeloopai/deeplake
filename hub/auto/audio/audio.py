import os

import hub
import numpy as np
from hub.auto import util
from hub.auto.infer import state
import librosa
from tqdm import tqdm

USE_TQDM = True


@state.directory_parser(priority=2)
def data_from_audio(path, scheduler, workers):

    try:
        import librosa
    except ModuleNotFoundError:
        raise ModuleNotInstalledException("librosa")

    if not util.files_are_of_extension(path, AUDIO_EXTS):
        return None

    max_shape = max_shape = (1920000,)
    filepaths = util.get_children(child)

    for filepath in filepaths:
        if util.get_ext(filepath) not in AUDIO_EXTS:
            continue

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

    # create transform for putting data into hub format
    @hub.transform(schema=schema, scheduler=scheduler, workers=workers)
    def upload_data(df):
        for index, row in df.iterrows():
            # for i in tqdm(range(df.columns)):
            #   ds["audio", i] = df["Audio"][i]
            #   ds["sampling_rate", i] = images_df["Sample Rate"][i]
            return {"image": index, "sampling_rate": row}

        return upload_data(data)
