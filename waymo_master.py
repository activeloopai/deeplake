import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
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
from hub.backend.storage import S3 as S3, GZipStorage
from PIL import Image
from time import clock
from pathos.multiprocessing import ProcessPool, ThreadPool
import waymo_upload

def frames_tfrecord(filepath):
    frame_count = 0
    dataset = tf.data.TFRecordDataset(filepath)
    for data in dataset:
        frame_count += 1

    return frame_count

path = '/home/edward/waymo/validation/'
filenames = os.listdir(path)
filenames.sort()
pool = ProcessPool(16)
data = pool.map(frames_tfrecord, map(lambda f: path + f, filenames))
frames = sum(data, 0)
print('Frames in files: {}, Total: {}'.format(data, frames))

start_frame = []
for i in range(0, frames):
    start_frame.append(sum(data[:i],0))
dataset_type = 'validation'
version = 'v2'
storage = S3(bucket='waymo-dataset-upload')
labels_arr = hub.array(shape=(frames, 2, 6, 30, 7), chunk_size=(100, 2, 6, 30, 7), storage=storage, name='edward/{}-labels:{}'.format(dataset_type, version), backend='s3', dtype='float64')
images_arr = hub.array(compression='jpeg', shape=(frames, 6, 1280, 1920, 3), storage=storage, name='edward/{}-camera-images:{}'.format(dataset_type, version), backend='s3', dtype='uint8', chunk_size=(1, 6, 1280, 1920, 3))    


def upload_record(i):
    waymo_upload.upload_tfrecord(dataset_type, path + filenames[i], version, start_frame[i])
    # os.system('python3 -c "from waymo_upload import upload_the_record; upload_the_record() ", {} {} {} {}'.format(dataset_type, path + filenames[i], version, start_frame[i]))

# for i in range(0, len(filenames), 2):
#     upload_record(i)
#     print("Finished {}".format(filenames[i]))

print("Second stage")

# for i in range(1, len(filenames), 2):
#     upload_record(i)
#     print("Finished {}".format(filenames[i]))


for i in range(0, 5):
    print("Stage {}".format(i))
    pool.map(upload_record, range(i, len(filenames), 5))

