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

filenames = os.listdir('/home/edward/waymo/validation')[0:4]
filenames = ['/home/edward/waymo/validation/' + filename for filename in filenames]

frame_count = 0
for filename in filenames:
    dataset = tf.data.TFRecordDataset(filename)
    for data in dataset:
        frame_count += 1

print(frame_count)

images_arr = hub.array(shape=(frame_count, 6, 1280, 1920, 3), name='edward/waymo-camera-images:v3', backend='s3', dtype='uint8', chunk_size=(1, 3, 1280, 1920, 3))

frames = frame_count
frame_count = 0
file_count = 0
for filename in filenames:
    file_count += 1
    dataset = tf.data.TFRecordDataset(filename)
    for batch in dataset.batch(20):
        l = batch.shape[0]
        for i in range(0, l):
            data = batch[i]
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            arr = np.zeros(shape=(l, 6, 1280, 1920, 3))
            for image in frame.images:
                img = np.array(Image.open(io.BytesIO(bytearray(image.image))))
                print('{} {}'.format(image.name, img.shape))
                arr[i, image.name, :img.shape[0], :img.shape[1]] = img
    
        images_arr[frame_count:frame_count+l] = arr
        frame_count += l
        print('Frames: {} Files: {} Total frames {}'.format(frame_count, file_count, frames))    






