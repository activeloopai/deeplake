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
    str_lasers_range_image = 'edward/{}-lasers-range-image:{}'.format(dataset_type, version)
    str_lasers_range_image_first = 'edward/{}-lasers-range-image-first:{}'.format(dataset_type, version)
    str_lasers_camera_proj = 'edward/{}-lasers-camera-proj:{}'.format(dataset_type, version)
    str_lasers_camera_proj_first = 'edward/{}-lasers-camera-proj-first:{}'.format(dataset_type, version)
    hub_lasers_range_image = hub.load(name=str_lasers_range_image, storage=storage)
    hub_lasers_camera_proj = hub.load(name=str_lasers_camera_proj, storage = storage)
    hub_lasers_range_image_first = hub.load(name=str_lasers_range_image_first, storage=storage)
    hub_lasers_camera_proj_first = hub.load(name=str_lasers_camera_proj_first, storage=storage)

    dataset = tf.data.TFRecordDataset(filepath)

    for batch in dataset.batch(1):
        def get_arr_image(range_image_compressed):
           data = zlib.decompress(range_image_compressed)
           mt = open_dataset.MatrixFloat()
           mt.ParseFromString(data)
           arr = np.reshape(np.array(mt.data), tuple(mt.shape.dims), order='C')
           return arr

        def get_arr_proj(camera_projection_compressed):
            data = zlib.decompress(camera_projection_compressed)
            mt = open_dataset.MatrixInt32()
            mt.ParseFromString(data)
            arr = np.reshape(np.array(mt.data), tuple(mt.shape.dims), order='C')
            return arr

        def get_frame_data(data):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            arr_image = np.zeros(shape=(4, 2, 200, 600, 4), dtype='float32')
            arr_proj = np.zeros(shape=(4, 2, 200, 600, 6), dtype='int32')
            arr_image_first = np.zeros(shape=(2, 64, 2650, 4), dtype='float32')
            arr_proj_first = np.zeros(shape=(2, 64, 2650, 6), dtype='int32')

            for laser in frame.lasers:
                laserri = [laser.ri_return1, laser.ri_return2]
                for i in range(0, 2):
                    if laser.name >= 2:
                        arr_image[laser.name - 2, i] = get_arr_image(laserri[i].range_image_compressed)
                        arr_proj[laser.name - 2, i] = get_arr_proj(laserri[i].camera_projection_compressed)
                    elif laser.name == 1:
                        arr_image_first[i] = get_arr_image(laserri[i].range_image_compressed)
                        arr_proj_first[i] = get_arr_proj(laserri[i].camera_projection_compressed) 
            return (arr_image, arr_proj, arr_image_first, arr_proj_first)
                
        l = int(batch.shape[0])
        for i in range(0, l):
            j = start_frame + i
            a, b, c, d = get_frame_data(batch[i])
            hub_lasers_range_image[j] = a
            hub_lasers_camera_proj[j] = b
            hub_lasers_range_image_first[j] = c
            hub_lasers_camera_proj_first[j] = d 

        start_frame += l

def main():
    path = '/home/edward/waymo/training/'
    dataset_type = 'training'
    version = 'v2'
    filenames = os.listdir(path)
    filenames.sort()
    pool = ProcessPool(16)
    frame_count_arr = pool.map(frames_tfrecord, map(lambda f: path + f, filenames))
    frames = sum(frame_count_arr, 0)

    str_lasers_range_image = 'edward/{}-lasers-range-image:{}'.format(dataset_type, version)
    str_lasers_range_image_first = 'edward/{}-lasers-range-image-first:{}'.format(dataset_type, version)
    str_lasers_camera_proj = 'edward/{}-lasers-camera-proj:{}'.format(dataset_type, version)
    str_lasers_camera_proj_first = 'edward/{}-lasers-camera-proj-first:{}'.format(dataset_type, version)
    storage = S3(bucket='waymo-dataset-upload')
    hub.array(shape=(frames, 4, 2, 200, 600, 4), dtype='float32', backend='s3', storage=storage, name=str_lasers_range_image, chunk_size=(1, 4, 2, 200, 600, 4))
    hub.array(shape=(frames, 4, 2, 200, 600, 6), dtype='int32', backend='s3', storage=storage, name=str_lasers_camera_proj, chunk_size=(1, 4, 2, 200, 600, 6))
    hub.array(shape=(frames, 2, 64, 2650, 4), dtype='float32', backend='s3', storage=storage, name=str_lasers_range_image_first, chunk_size=(1, 2, 64, 2650, 4))
    hub.array(shape=(frames, 2, 64, 2650, 6), dtype='int32', backend='s3', storage=storage, name=str_lasers_camera_proj_first, chunk_size=(1, 2, 64, 2650, 6))

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