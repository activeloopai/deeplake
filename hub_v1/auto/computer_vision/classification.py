import os

import hub_v1
import numpy as np
from hub_v1.auto import util
from hub_v1.auto.infer import state
from PIL import Image
from tqdm import tqdm

USE_TQDM = True


@state.directory_parser(priority=0)
def image_classification(path, scheduler, workers):
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
    max_shape = np.zeros(3, dtype=np.uint8)  # CHW
    all_same_shape = True
    image_shape = None  # if shape is all the same, use that instead of max_shape
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

            shape = np.array(util.get_image_shape(filepath))

            # check if all images have the same shape
            if all_same_shape:
                if image_shape is None:
                    image_shape = shape
                elif not np.array_equal(image_shape, shape):
                    all_same_shape = False

            max_shape = np.maximum(max_shape, shape)
            data.append((filepath, label.lower()))
            class_names.add(label.lower())

    # create schema
    class_names = list(sorted(list(class_names)))
    max_shape = tuple([int(x) for x in max_shape])
    actual_shape = (
        tuple([int(x) for x in image_shape]) if all_same_shape else (None, None, None)
    )
    max_shape = None if all_same_shape else max_shape
    schema = {
        "image": hub_v1.schema.Image(
            shape=actual_shape, dtype="uint8", max_shape=max_shape
        ),
        "label": hub_v1.schema.ClassLabel(names=class_names),
    }

    # create transform for putting data into hub format
    @hub_v1.transform(schema=schema, scheduler=scheduler, workers=workers)
    def upload_data(sample):
        path = sample[0]
        img = Image.open(path)
        img = np.asarray(img)
        label = class_names.index(sample[1])
        return {"image": img, "label": label}

    return upload_data(data)
