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
from hub.backend.storage import S3

# filenames = os.listdir('/home/edward/waymo/validation')[0:2]
# filenames = ['/home/edward/waymo/validation/' + filename for filename in filenames]
# print(filenames)
# frame_count = 0

# file_counter = 0
# for filename in filenames:
#     dataset = tf.data.TFRecordDataset(filename)
#     # print('Frame: {}, File: {}'.format(frame_count, file_counter))
#     file_counter += 1
#     for data in dataset:
#         frame_count += 1
#         # frame = open_dataset.Frame()
#         # frame.ParseFromString(bytearray(data.numpy()))

# print('Total frames: {}'.format(frame_count))
# print('Total files: {}'.format(file_counter))

# arr = hub.array((frame_count, 6 * 1000 * 1000), name='edward/waymo-dataset:v2', backend='s3', dtype='uint8', chunk_size=(6, 6 * 1000 * 1000))

# counter = 0
# for filename in filenames:
#     dataset = tf.data.TFRecordDataset(filename)
#     for batch in dataset.batch(6):
#         x = np.zeros(shape=(batch.shape[0], 6 * 1000 * 1000), dtype='uint8')
#         print(batch.shape[0])
#         for i in range(0, batch.shape[0]):
#             data = bytearray(batch[i].numpy())
#             l = len(data)
#             x[i][4:4 + l] = data
#             x[i][0] = l // 256 // 256 // 256
#             x[i][1] = (l // 256 // 256) % 256
#             x[i][2] = (l // 256) % 256
#             x[i][3] = l % 256
#         arr[counter:counter + batch.shape[0]] = x
#         counter += batch.shape[0]

# starting reading data to check integrity

# arr2 = hub.load(name='edward/waymo-dataset:v2', backend='s3')

# data = arr2[108]

# l = data[0] * 256 * 256 * 256 + data[1] * 256 * 256 + data[2] * 256 + data[3]
# x = data[4:4 + l]

# frame = open_dataset.Frame()
# frame.ParseFromString(bytearray(x))
# image = frame.images[0]
# img = Image.open(io.BytesIO(bytearray(image.image)))
# print(np.array(img).shape)

# arr = hub.load(name='edward/validation-camera-images:v2', backend='s3', storage=S3(bucket='waymo-dataset-upload'))

# img = arr[200][1]
# Image.fromarray(img, 'RGB').save('image.png')

# arr = hub.array(shape=(100, 100, 100000), name='test6', chunk_size=(1, 1, 100000), storage=S3(bucket='waymo-dataset-upload'), compression='gzip')
# arr[5,5] = np.ones(shape=(100000))
# arr[6,6] = np.ones(shape=(100000))
# print(arr[5, 5, 3])

bucket = hub.S3(bucket='waymo-dataset-upload')
arr = bucket.arraynew(shape=(5, 1920, 1080, 3), name='test10', dtype='uint8', chunk=(1, 1920, 1080, 3), compress='jpeg')
arr[2] = np.ones((1920, 1080, 3))
arr[3] = np.ones((1920, 1080, 3))
print(arr[3, 120, 120, 2])

# arr = np.array([[1, 2], [3, 4]])
# print(type(arr))
# print(arr[1])
# print(type(arr[1]))