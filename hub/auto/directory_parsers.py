import os
from tqdm import tqdm
import numpy as np
from PIL import Image

import hub

from hub.auto import util

USE_TQDM = True


@util.directory_parser(priority=0)
def image_classification(path, scheduler='single', workers=1):
    children = util.get_children(path)

    # check if all children are directories
    if not all([os.path.isdir(child) for child in children]):
        return None

    # check if children's contents are all image files
    for child in children:
        if not util.files_are_of_extension(child, util.IMAGE_EXTS):
            return None

    # parse dataset
    data = []
    class_names = set()
    max_shape = np.zeros(3, dtype=np.uint8)  # CHW
    for child in tqdm(children,
                      desc='parsing image classification dataset',
                      total=len(children),
                      disable=not USE_TQDM):
        label = os.path.basename(child)

        filepaths = util.get_children(child)
        for filepath in filepaths:
            # ignore non-image extension files
            if util.get_ext(filepath) not in util.IMAGE_EXTS:
                continue

            shape = np.array(util.get_image_shape(filepath))
            max_shape = np.maximum(max_shape, shape)
            data.append((filepath, label.lower()))
            class_names.add(label.lower())

    class_names = list(sorted(list(class_names)))

    # create schema
    max_shape = tuple([int(x) for x in max_shape])
    schema = {
        'image':
        hub.schema.Image(shape=(None, None, None),
                         dtype='uint8',
                         max_shape=max_shape),
        'label':
        hub.schema.ClassLabel(names=class_names)
    }

    # create transform for putting data into hub format
    @hub.transform(schema=schema, scheduler=scheduler, workers=workers)
    def upload_data(sample):
        path = sample[0]
        img = Image.open(path)
        img = np.asarray(img)
        label = class_names.index(sample[1])
        return {'image': img, 'label': label}

    return upload_data(data)
