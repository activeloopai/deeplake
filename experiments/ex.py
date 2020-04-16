import os
# import tensorflow as tf
import math
import numpy as np
import itertools
import io

# tf.enable_eager_execution()

# from waymo_open_dataset.utils import range_image_utils
# from waymo_open_dataset.utils import transform_utils
# from waymo_open_dataset.utils import  frame_utils
# from waymo_open_dataset import dataset_pb2 as open_dataset
import hub
from PIL import Image

# client = hub.gs('snark_waymo_open_dataset', creds_path='.creds/gs.json').connect() 
client = hub.fs('/drive/upload').connect()
arr = client.array_open('validation/images')

for i in range(0, 5):
    img = arr[10, i]
    print(img.shape)
    Image.fromarray(img, 'RGB').save(f'output/image-{i}.jpg')

