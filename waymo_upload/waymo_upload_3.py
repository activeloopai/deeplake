import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import math
import numpy as np
import itertools
import io
import sys

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import hub
import zlib
from hub.backend.storage import S3 as S3, GZipStorage
from PIL import Image
from time import clock
from pathos.multiprocessing import ProcessPool

def frames_tfrecord(filepath):
    frame_count = 0
    dataset = tf.data.TFRecordDataset(filepath)
    for data in dataset:
        frame_count += 1

    return frame_count

def upload_tfrecord(dataset_type, filepath, version, start_frame):
    storage = S3(bucket='waymo-dataset-upload')
    # str_lasers_range_image = 'edward/{}-lasers-range-image:{}'.format(dataset_type, version)
    # str_lasers_range_image_first = 'edward/{}-lasers-range-image-first:{}'.format(dataset_type, version)
    # str_lasers_camera_proj = 'edward/{}-lasers-camera-proj:{}'.format(dataset_type, version)
    # str_lasers_camera_proj_first = 'edward/{}-lasers-camera-proj-first:{}'.format(dataset_type, version)
    # hub_lasers_range_image = hub.load(name=str_lasers_range_image, storage=storage)
    # hub_lasers_camera_proj = hub.load(name=str_lasers_camera_proj, storage = storage)
    # hub_lasers_range_image_first = hub.load(name=str_lasers_range_image_first, storage=storage)
    # hub_lasers_camera_proj_first = hub.load(name=str_lasers_camera_proj_first, storage=storage)

    str_labels_laser = 'edward/{}-labels-laser:{}'.format(dataset_type, version)
    hub_labels_laser = hub.load(name=str_labels_laser, storage=storage)

    dataset = tf.data.TFRecordDataset(filepath)

    for batch in dataset.batch(100):
        def get_frame_data(data):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            arr_labels_laser = np.zeros(shape=(1, 30, 7), dtype='float64')

            for label in frame.laser_labels:
                box = label.box
                arr_labels_laser = np.array([box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, box.heading])
            return arr_labels_laser
                
        l = int(batch.shape[0])
        arr = np.zeros(shape=(l, 30, 7), dtype='float64')
        for i in range(0, l):
            j = start_frame + i
            arr[i] = get_frame_data(batch[i])

        hub_labels_laser[start_frame:start_frame+l] = arr
        start_frame += l

def main():
    path = '/home/edward/waymo/validation/'
    dataset_type = 'validation'
    version = 'v2'
    filenames = os.listdir(path)
    filenames.sort()
    pool = ProcessPool(16)
    frame_count_arr = pool.map(frames_tfrecord, map(lambda f: path + f, filenames)) 
    frames = sum(frame_count_arr, 0)

    str_labels_laser = 'edward/{}-labels-laser:{}'.format(dataset_type, version)
    storage = S3(bucket='waymo-dataset-upload')

    hub.array(shape=(frames, 30, 7), dtype='float64', storage=storage, chunk_size=(100, 30, 7), name=str_labels_laser)

    # str_lasers_range_image = 'edward/{}-lasers-range-image:{}'.format(dataset_type, version)
    # str_lasers_range_image_first = 'edward/{}-lasers-range-image-first:{}'.format(dataset_type, version)
    # str_lasers_camera_proj = 'edward/{}-lasers-camera-proj:{}'.format(dataset_type, version)
    # str_lasers_camera_proj_first = 'edward/{}-lasers-camera-proj-first:{}'.format(dataset_type, version)
    # storage = S3(bucket='waymo-dataset-upload')
    # hub.array(shape=(frames, 4, 2, 200, 600, 4), dtype='float32', backend='s3', storage=storage, name=str_lasers_range_image, chunk_size=(1, 4, 2, 200, 600, 4))
    # hub.array(shape=(frames, 4, 2, 200, 600, 6), dtype='int32', backend='s3', storage=storage, name=str_lasers_camera_proj, chunk_size=(1, 4, 2, 200, 600, 6))
    # hub.array(shape=(frames, 2, 64, 2650, 4), dtype='float32', backend='s3', storage=storage, name=str_lasers_range_image_first, chunk_size=(1, 2, 64, 2650, 4))
    # hub.array(shape=(frames, 2, 64, 2650, 6), dtype='int32', backend='s3', storage=storage, name=str_lasers_camera_proj_first, chunk_size=(1, 2, 64, 2650, 6))

    start_frame_arr = []
    for i in range(0, len(filenames)):
        start_frame_arr.append(sum(frame_count_arr[:i], 0))

    def upload(i):
        upload_tfrecord(dataset_type, path + filenames[i], version, start_frame_arr[i])

    # upload(0)

    for i in range(0, 5):
        pool.map(upload, range(i, len(filenames), 5))
        print('Stage {} finished'.format(i))

if __name__ == '__main__':
    main()