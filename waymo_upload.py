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
from hub.backend.storage import S3 as S3, GZipStorage
from PIL import Image
from time import clock
from pathos.multiprocessing import ProcessPool

# filenames = ['/home/edward/waymo/validation/' + filename for filename in filenames]

def frames_tfrecord(filepath):
    frame_count = 0
    dataset = tf.data.TFRecordDataset(filepath)
    for data in dataset:
        frame_count += 1

    return frame_count

def upload_tfrecord(dataset_type, filepath, version, start_frame):
    storage = S3(bucket='waymo-dataset-upload')
    label_name = 'edward/{}-labels:{}'.format(dataset_type, version)
    image_name = 'edward/{}-camera-images:{}'.format(dataset_type, version)
    # print('{} {}'.format(label_name, image_name))
    images_arr = hub.load(name=image_name, storage=storage)
    labels_arr = hub.load(name=label_name, storage=storage)
    frame_count = start_frame
    dataset = tf.data.TFRecordDataset(filepath)
    # print('Yeah {}'.format(frame_count))
    for batch in dataset.batch(16):
        # print('Cycle')
        t1 = clock()
        l = batch.shape[0]
        arr = np.zeros(shape=(l, 6, 1280, 1920, 3), dtype='uint8')
        lab = np.zeros(shape=(l, 2, 6, 30, 7), dtype='float64')
        for i in range(0, l):
            # print('Cycle2')
            data = batch[i]
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            for image in frame.images:
                # print('Cycle3')
                img = np.array(Image.open(io.BytesIO(bytearray(image.image))))
                arr[i, image.name, :img.shape[0], :img.shape[1]] = img
            llx = [frame.projected_lidar_labels, frame.camera_labels]
            li = 0
            for ll in llx:
                # print('Cycle4')
                for labels in ll:
                    name = labels.name
                    j = 0
                    for label in labels.labels:
                        box = label.box
                        x = np.array([box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, box.heading])
                        lab[i, li, name, j] = x
                        j += 1
                        if j == 30:
                            break
                li += 1
        # print('Cycle5')
        t2 = clock()    
        # print('Cycle6')
        # print(arr[5,5])
        images_arr[frame_count:frame_count+l] = arr
        # print('Cycle7')
        labels_arr[frame_count:frame_count+l] = lab
        # print('Cycle8')
        t3 = clock()
        frame_count += l
        # print('{} : {}', t3 - t2, t3 - t1) 
    print('{} finished'.format(filepath))




def upload_all():
    path = '/home/edward/waymo/training/'
    filenames = os.listdir(path)
    filenames.sort()
    pool = ProcessPool(16)
    data = pool.map(frames_tfrecord, map(lambda f: path + f, filenames))
    frames = sum(data, 0)
    print('Frames in files: {}, Total: {}'.format(data, frames))

    start_frame = []
    for i in range(0, frames):
        start_frame.append(sum(data[:i],0))
    dataset_type = 'training'
    version = 'v2'
    storage = S3(bucket='waymo-dataset-upload')
    labels_arr = hub.array(shape=(frames, 2, 6, 30, 7), chunk_size=(100, 2, 6, 30, 7), storage=storage, name='edward/{}-labels:{}'.format(dataset_type, version), backend='s3', dtype='float64')
    images_arr = hub.array(compression='jpeg', shape=(frames, 6, 1280, 1920, 3), storage=storage, name='edward/{}-camera-images:{}'.format(dataset_type, version), backend='s3', dtype='uint8', chunk_size=(1, 6, 1280, 1920, 3))    

    def upload_record(i):
        upload_tfrecord(dataset_type, path + filenames[i], version, start_frame[i])

    for i in range(0, 5):
        print("Stage {}".format(i))
        pool.map(upload_record, range(i, len(filenames), 5))

if __name__ == '__main__':
    upload_all()