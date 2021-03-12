import hub
import tensorflow as tf
import numpy as np
from PIL import Image


def only_frontal(sample):
    viewPosition = sample["viewPosition"].compute(True)
    return True if "PA" in viewPosition or "AP" in viewPosition else False


def get_image(viewPosition, images):
    for i, vp in enumerate(viewPosition):
        if vp in [5, 12]:
            return np.concatenate((images[i], images[i], images[i]), axis=2)


def to_model_fit(sample):
    viewPosition = sample["viewPosition"]
    images = sample["image"]
    image = tf.py_function(get_image, [viewPosition, images], tf.uint16)
    labels = sample["label_chexpert"]
    return image, labels


ds = hub.Dataset(
    "s3://snark-gradient-raw-data/output_single_8_5000_samples_max_4_boolean_m5_fixed/ds3")
dsf = ds.filter(only_frontal)
tds_train = dsf.to_tensorflow(
    key_list=["image", "label_chexpert", "viewPosition"])
tds_train = tds_train.map(to_model_fit)
tds_train = tds_train.batch(8).prefetch(tf.data.AUTOTUNE)
for i, item in enumerate(tds_train):
    if i%5 == 0:
        print("saving")
        im = Image.fromarray(255 * item[0][0].numpy().astype("uint8"))
        im.save(f"./img/{i//5}.jpeg")
    # print(item[0][0].numpy())
    # if i<10:
    #     im = Image.fromarray(item[0][0].numpy())
        # im.save(f"./img/{i%50}.jpeg")

    # print(item[0][:, 300:314, 2, 1])
