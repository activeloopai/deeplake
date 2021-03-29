import os

import hub
import numpy as np
from hub.auto import util
from hub.auto.infer import state
from PIL import Image
from tqdm import tqdm

USE_TQDM = True


@state.directory_parser(priority=0)
def image_classification(path, config):
    children = util.get_children(path, only_dirs=True)

    # check if there is >= 2 children (means there are at least 2 folders)
    if len(children) < 2:
        print("test1")
        return None

    # check if children's contents has image files
    for child in children:
        if not util.files_are_of_extension(child, util.IMAGE_EXTS):
            print("test2")
            return None

    # parse dataset
    data = []
    class_names = set()
    for child in tqdm(
        children,
        desc="parsing image classification dataset",
        total=len(children),
        disable=not USE_TQDM,
    ):
        label = os.path.basename(child)

        filepaths = util.get_children(child)
        for filepath in filepaths:
            # ignore non-image extension files
            if util.get_ext(filepath) not in util.IMAGE_EXTS:
                continue

            data.append((filepath, label.lower()))
            class_names.add(label.lower())

    # create schema
    class_names = list(sorted(list(class_names)))
    shape, max_shape = util.infer_shape(path, p=config["p"], use_tqdm=USE_TQDM)
    schema = {
        "image": hub.schema.Image(
            shape=shape, dtype="uint8", max_shape=max_shape
        ),
        "label": hub.schema.ClassLabel(names=class_names),
    }

    # create transform for putting data into hub format
    @hub.transform(schema=schema, scheduler=config["scheduler"], 
                   workers=config["workers"])
    def upload_data(sample):
        path = sample[0]
        img = Image.open(path)
        img = np.asarray(img)
        label = class_names.index(sample[1])
        return {"image": img, "label": label}

    return upload_data(data)
