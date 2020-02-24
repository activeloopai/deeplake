import hub
import numpy as np
import sys, os, time, random, uuid, itertools, json, traceback, io

# Create
conn = hub.s3(
    'waymo-dataset-upload', 
    aws_creds_filepath='.creds/aws.json'
    ).connect()


x = conn.array_create(
    shape = (50000, 250, 250, 3),
    chunk=(4, 250, 250, 3),
    name= os.path.join('test', f'{int(time.time())}'),
    dtype='uint8',
)

# Upload
x[0] = np.ones((250, 250, 3), dtype='uint8')

# Download
print(x[0].mean())
print(x[1].mean())
print(x.shape, x[1].shape)
