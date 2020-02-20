from typing import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import math
import numpy as np
import itertools
import io
import sys
import zlib

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
from pathos.pools import SerialPool
import multiprocessing as mp
from functools import reduce
import time
import gc
from concurrent.futures import ProcessPoolExecutor

def conn_setup(bucket: str = 'waymo-dataset-upload') -> hub.Bucket:
    return hub.s3(bucket=bucket, aws_access_key_id='AKIAIUUHCNWTJRL3MLDA', aws_secret_access_key='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').connect()

class Waymo:
    dir_in: str = None
    dir_out: str = None
    batch_size: int = 0
    pool_size: int = 0
    debug: int = None

    def __init__(self, dir_in: str = '~/waymo/', dir_out: str = 'v029'):
        self.dir_in = os.path.expanduser(dir_in)
        self.dir_out = dir_out
        self.batch_size = 1
        self.pool_size = 8
        self.debug = None

    def get_filenames(self, tag: str):
        dir = os.path.join(self.dir_in, tag)
        files = os.listdir(dir)
        files.sort()
        files = [os.path.join(dir, f) for f in files]  
        # files = ['/home/edward/waymo/validation/segment-3126522626440597519_806_440_826_440.tfrecord']
        if self.debug is not None:
            return list(files)[:self.debug]
        else:
            return list(files)
    
    def get_frames(self, files: [str]):
        # ctx = mp.get_context('fork')
        frames = None

        with ProcessPoolExecutor(self.pool_size) as pool:
            frames = pool.map(self.frame_count_file, files)  
        # pool.terminate()

        if self.debug is not None:
            return list(frames)[:self.debug]
        else:
            return list(frames)    

    def frame_count_file(self, filepath: str) -> int:
        dataset = tf.data.TFRecordDataset(filepath)
        return sum(1 for _ in dataset)

    def create_dataset(self, tag: str, frame_count: int) -> hub.Dataset:
        bucket = conn_setup()
        dataset = {}
        info = {
            'labels': {
                'shape': (frame_count, 11, 400, 7),
                'chunk': (190, 11, 400, 7),
                'dtype': 'float64',
                'compress': 'lz4',
                'dsplit': 2,
            },

            'lasers_camera_projection': {
                'shape': (frame_count, 5, 2, 200, 2650, 6),
                'chunk': (self.batch_size, 5, 2, 200, 2650, 6),
                'dtype': 'int32',
                'compress': 'lz4',
                'dsplit': 3,
            },

            'images': {
                'shape': (frame_count, 5, 1280, 1920, 3),
                'chunk': (self.batch_size, 5, 1280, 1920, 3),
                'dtype': 'uint8',
                'compress': 'jpeg',
                'dsplit': 2,
            },

            'lasers_range_image': {
                'shape': (frame_count, 5, 2, 200, 2650, 4),
                'chunk': (self.batch_size, 5, 2, 200, 2650, 4),
                'dtype': 'float32',
                'compress': 'lz4',
                'dsplit': 3,
            },
        }

        for key, value in info.items():
            dataset[key] = bucket.array_create(os.path.join(self.dir_out, tag, key), **value)
        
        ds = bucket.dataset_create(os.path.join(self.dir_out, tag, 'dataset'), dataset)
        ds['images'].darray[:, :3] = (1280, 1920, 3)
        ds['images'].darray[:,3:] = (886, 1920, 3)
        ds['lasers_range_image'].darray[:, 0:1, :] = (64, 2650, 4)
        ds['lasers_range_image'].darray[:, 1:, :] = (200, 600, 4)
        ds['lasers_camera_projection'].darray[:, 0:1, :] = (64, 2650, 6)
        ds['lasers_camera_projection'].darray[:, 1:, :] = (200, 600, 6)
        return ds

    def extract_record_batch(self, ds: hub.Dataset, arr: Dict[str, np.ndarray], index: int, batch) -> int:
        cnt = 0
        for key in ['images', 'lasers_range_image', 'lasers_camera_projection']:
            arr[key] = np.zeros(ds[key].chunk, ds[key].dtype)

        for i, sample in enumerate(batch):
            t1 = time.time()
            cnt += 1
            
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(sample.numpy()))
            
            key = 'images'
            arrkey = arr[key]
            for image in frame.images:
                if image.name > 0:
                    img = np.array(Image.open(io.BytesIO(bytearray(image.image))))
                    arrkey[i, image.name - 1, :img.shape[0], :img.shape[1]] = img
                     

            arrkey = arr['labels'][index + i]
            for j, key in enumerate(['camera_labels', 'projected_lidar_labels']):    
                for image in getattr(frame, key):
                    if image.name > 0:
                        _count = 0
                        for l, label in enumerate(image.labels):
                            box = label.box
                            x = np.array([box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, box.heading])
                            arrkey[5 * j + image.name, l] = x
                            _count += 1
                        
                        ds['labels'].darray[index + i][5 * j + image.name] = (_count, 7)

            
            key = 'laser_labels'
            _count = 0
            for j, label in enumerate(getattr(frame, key)):
                box = label.box
                x = np.array([box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, box.heading])
                arrkey[0, j] = x
                _count += 1
            
            ds['labels'].darray[index + i][0] = (_count, 7)

            for kid, key in enumerate(['lasers_range_image', 'lasers_camera_projection']):
                arrkey = arr[key]
                for laser in frame.lasers:
                    if laser.name > 0:
                        for j, ri in enumerate([laser.ri_return1, laser.ri_return2]):
                            attr_name = '_'.join(key.split('_')[1:]) + '_compressed'
                            data = getattr(ri, attr_name)
                            mt = open_dataset.MatrixFloat() if kid == 0 else open_dataset.MatrixInt32()
                            mt.ParseFromString(zlib.decompress(data))
                            
                            _arr = np.reshape(
                                np.array(mt.data), 
                                tuple(mt.shape.dims),
                                ) 

                            arrkey[i, laser.name - 1, j, :_arr.shape[0], :_arr.shape[1]] = _arr
                            # print(f'{laser.name - 1}, {key}: {_arr.shape}')
            t2 = time.time()

        return cnt

    def upload_record_file(self, filepath: str, tag: str, index: int):
        _t1 = time.time()
        ds = conn_setup().dataset_open(os.path.join(self.dir_out, tag, 'dataset'))
        waymo = tf.data.TFRecordDataset(filepath)
        arr = {}
        key = 'labels'
        arr[key] = np.zeros((256,) + tuple(ds[key].chunk[1:]), ds[key].dtype)

        def batched(waymo, sz: int):
            batch = []
            for sample in waymo:
                batch.append(sample)
                if len(batch) == sz:
                    yield batch
                    batch = []
            
            if len(batch) > 0:
                yield batch

        cnt = 0
        for batch in batched(waymo, self.batch_size):
            t1 = time.time()
            sys.stdout.flush()
            gc.collect()
            _cnt = self.extract_record_batch(ds, arr, cnt, batch)
        
            for key, _ in ds.items():
                if key != 'labels':
                    ds[key][index + cnt : index + cnt + _cnt] = arr[key][:_cnt]
            
            cnt += _cnt
            t2 = time.time()
            # print('Batch finished in {}s'.format(t2 - t1))

        key = 'labels'
        ds[key][index : index + cnt] = arr[key][:cnt]
        
        _t2 = time.time()
        print('record {} uploaded in {}s'.format(filepath, _t2 - _t1))

    def _upload_record_file(self, args):
        self.upload_record_file(*args)
                
    def upload_record_tag(self, tag: str, frames: [int]):
        files = self.get_filenames(tag)
        # frames = self.get_frames(tag)
        frames = [0] + frames[:-1]

        for i in range(1, len(frames)):
            frames[i] += frames[i - 1]

        assert len(frames) == len(files)

        # ctx = mp.get_context('spawn')

        print(frames)
        sys.stdout.flush()

        with ProcessPoolExecutor(self.pool_size) as pool:
            input = list(zip(files, [tag] * len(files), frames))

            for i in range(0, 2):
                list(pool.map(self._upload_record_file, [input[x] for x in range(i, len(input), 2)]))
        # pool.terminate()
        
    def process(self):
        for tag in ['validation', 'training']:
            t1 = time.time()
            print('Total {} files in {}'.format(len(self.get_filenames(tag)), tag))
            files = self.get_filenames(tag)
            frames = self.get_frames(self.get_filenames(tag))
            print('Total {} frames in {}'.format(sum(frames), tag))
            self.create_dataset(tag, sum(frames))
            self.upload_record_tag(tag, frames)
            t2 = time.time()
            print('Finishing tag: {}, duration {}s'.format(tag, t2 - t1))


if __name__ == '__main__':
    print('Hello World')
    time.sleep(1)
    Waymo().process()