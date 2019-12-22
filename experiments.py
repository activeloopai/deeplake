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

# import os
# import tensorflow as tf
# import math
# import numpy as np
# import itertools
# import io

# tf.enable_eager_execution()

# from waymo_open_dataset.utils import range_image_utils
# from waymo_open_dataset.utils import transform_utils
# from waymo_open_dataset.utils import  frame_utils
# from waymo_open_dataset import dataset_pb2 as open_dataset
# import hub
# from PIL import Image

# filenames = os.listdir('/home/edward/waymo/validation')[0:1]
# filenames = ['/home/edward/waymo/validation/' + filename for filename in filenames]
# print(filenames)
# image_count = 0
# frame_count = 0

# file_counter = 0
# for filename in filenames:
#     try:
#         dataset = tf.data.TFRecordDataset(filename)
#         print('Image Count: {} File Count: {}/{}'.format(image_count, file_counter + 1, len(filenames)))
#         file_counter += 1
#         for data in dataset:
#             frame_count += 1
#             frame = open_dataset.Frame()
#             frame.ParseFromString(bytearray(data.numpy()))
#             for image in frame.images:
#                 if image.name == 1:
#                     image_count += 1 
#     except:
#         print(filename)


# arr = hub.array((image_count, 1280, 1920, 3 ), name='edward/waymo-images:v9', backend='s3', dtype='uint8', chunk_size=(3, 1280, 1920, 3))
# ds = hub.dataset(name = 'edward/waymo-images-dataset', arrays = {'images': arr})

# image_index = 0
# frame_index = 0
# for filename in filenames:
#     dataset = tf.data.TFRecordDataset(filename)
#     for datas in dataset.batch(30):
#         images = []
#         print('{}/{}'.format(frame_index + 1, frame_count))
#         for data in datas: 
#             frame_index += 1
#             frame = open_dataset.Frame()
#             frame.ParseFromString(bytearray(data.numpy()))
#             for image in frame.images:
#                 if image.name == 1:
#                     img = Image.open(io.BytesIO(bytearray(image.image)))
#                     a = np.array(img)
#                     print(a.shape)
#                     images.append(a)
#         arr[image_index:image_index + len(images)] = images
#         image_index += len(images)

# print(arr[6].shape)


# from hub.backend.storage import S3
# import numpy as np
# from time import clock
# from random import random

# storage = S3(bucket='waymo-dataset-upload')
# start = clock()
# storage.put('sample-{}.bin'.format(random()), np.zeros(shape=(4, 1000, 1000, 1000), dtype='uint8').tobytes())
# print(clock() - start)

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
import zlib
from PIL import Image


filenames = os.listdir('/home/edward/waymo/validation')[0:1]
filenames = ['/home/edward/waymo/validation/' + filename for filename in filenames]

frame_count = 0

for filename in filenames:
    dataset = tf.data.TFRecordDataset(filename)
    # print('Image Count: {} File Count: {}/{}'.format(image_count, file_counter + 1, len(filenames)))
    for data in dataset:
        frame_count += 1
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        for laser in frame.lasers:
            name = laser.name
            ri1 = laser.ri_return1
            ri2 = laser.ri_return2

            dec = zlib.decompress(ri1.range_image_compressed)
            range_image = open_dataset.MatrixFloat()
            range_image.ParseFromString(dec)
            print(name, range_image.shape.dims)