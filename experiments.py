# from google.cloud import storage
# client = storage.Client()
# buckets = list(client.list_buckets())
# my_bucket = client.get_bucket('snark_waymo_open_dataset')
# blob = my_bucket.blob('new_folder/temporary_blob.txt')
# blob.upload_from_string('Hello World This is lengthier')



# import hub

# my_array = hub.array(shape=(1000,), name='myarray:v0', backend='gs')
# my_array[5] = 6
# another_array = hub.load(name='myarray:v0', backend='gs')
# print(another_array[5])

# import tensorflow as tf 
# import os
# import itertools
# import functools

# # data = tf.data.TFRecordDataset(['waymo/' + filename for filename in os.listdir('waymo')])

# cnt = 0
# for filename in os.listdir('waymo'):
#     data = tf.data.TFRecordDataset('waymo/' + filename)
#     print(filename)
#     for _ in data:
#         cnt += 1

# print(cnt)

import os
import tensorflow as tf
import math
import numpy as np
import itertools
import io

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import hub
from PIL import Image

filename = 'waymo/' + os.listdir('waymo')[0]
dataset = tf.data.TFRecordDataset(filename)
cnt = 0

cnt2 = 0

for data in dataset:
    cnt2 += 1
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    for image in frame.images:
        if image.name == 1:
            cnt += 1 

arr = hub.array((cnt, 1280, 1920, 3 ), name='velocity:v0', backend='gs')
ds = hub.dataset(name = 'waymo_ds', arrays = {'images_velocity': arr})

i = 0
j = 0
for data in dataset:
    print("{}/{}".format(j, cnt2))
    j += 1
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    for image in frame.images:
        if image.name == 1:
            img = Image.open(io.BytesIO(bytearray(image.image)))
            a = np.array(img)
            print(a.shape)
            arr[i] = a
            print('Here')
            i += 1
            break



