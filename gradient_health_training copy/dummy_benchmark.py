
import hub
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
from collections import defaultdict
from hub.utils import Timer


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


def benchmark(dataset, num_epochs=1):
    # print("Length is", len(dataset))
    # start_time = time.perf_counter()
    with tqdm(
        total=len(dataset),
        unit_scale=True,
        unit=" items",
    ) as pbar:
        # ds = dataset.to_tensorflow(key_list=["image"]).prefetch(tf.data.AUTOTUNE)
        ds = dataset
        # dataset["image"].compute()
        # pbar.update(1000)
        # for i in range(10):
        #     ds["image", slice(i*100, i*100 + 100)].compute()
        #     pbar.update(100)
        for i in range(len(ds)):
            # print(np.max(sample["image"]))
            # for item in sample:
            #     x += item[0].shape[0]
            with Timer("ind"):
                sample = ds[i]
            with Timer("image"):
                a = sample["image"]
            with Timer("compute"):
                b = a.compute()
            # sample["label_chexpert"].compute()
            pbar.update(1)

            # x += sample["image"].shape[0]
            # print(i)
            # ct += 1
            # print("now sleeping")
            # time.sleep(0.05)
            # print("awake now")

            # Performing a training step
    # print(x/ct)
    # print("Execution time:", time.perf_counter() - start_time)


batch_size = 8
# ds = hub.Dataset("s3://snark-gradient-raw-data/output_ray_single_8_100k_2/ds3/")


# ds = hub.Dataset(
#     "s3://snark-gradient-raw-data/output_ray_single_8_1500_samples_chunked_100/ds3")

# ds = hub.Dataset("s3://snark-gradient-raw-data/output_ray_single_8_full_dataset/ds3")

# ds = hub.Dataset("s3://snark-gradient-raw-data/output_single_8_1500_samples_max_9/ds3")
ds = hub.Dataset(
    "s3://snark-gradient-raw-data/output_single_8_1500_samples_max_4_boolean/ds3/")
# ds = hub.Dataset(
#     "s3://snark-gradient-raw-data/output_single_8_1500_samples_max_4_boolean_no_comp_2/ds3")  # 14s

# ds = hub.Dataset(
#     "s3://snark-gradient-raw-data/output_single_8_1500_samples_max_4_boolean_chunks_100/ds3")
# ds = hub.Dataset(
#     "s3://snark-gradient-raw-data/output_single_8_1500_samples_max_4_boolean_no_comp_new/ds3")
# mx = 0
# d = defaultdict(int)
# av = 0
# with tqdm(
#     total=len(ds),
#     unit_scale=True,
#     unit=" items",
# ) as pbar:
#     for item in ds:
#         sh = item["image"].shape[0]
#         d[sh] += 1
#         mx = max(mx, sh)
#         av += sh

#         pbar.update(1)
#         # print(x)
# av = av/len(ds)
# print(d)
# print(mx)
# print(av)

# exit()

# a = ds["image", 0:5].compute()
# for item in a:
#     print(item.shape)
# print(ds["image", 0:5].compute())
# exit()
# print(ds.keys)
# 1743
# ds = hub.Dataset("s3://snark-gradient-raw-data/output_all_attributes_2500_samples_300_chunk/ds3")
dsv_train = ds[0:1000]
dsv_val = ds[1500:1864]
# dsv_val = ds[0:364]
# dsf_train = dsv_train.filter(only_frontal)
# dsf_val = dsv_val.filter(only_frontal)
# tds_train = dsf_train.to_tensorflow()
# tds_train = tds_train.map(to_model_fit)
# tds_train = tds_train.batch(batch_size)
# tds_val = dsf_val.to_tensorflow()
# tds_val = tds_val.map(to_model_fit)
# tds_val = tds_val.batch(batch_size)

benchmark(dsv_train)
# benchmark(dsf_val)
# benchmark(ds[0:1000])

# with tuning:-
# 20.517092499998398
# 18.888743067000178
# 17.347827763005625
# Execution time: 17.043490725991433


# only 2 tensors: label_chexpert and image. 1000 samples 26.63s
# only 1 tensor: image. 1000 samples 26.15603759500027
# only label 4.93s
# tds without key_list 50s
# with key_list 1000smp 23s 2000smp 47.23
