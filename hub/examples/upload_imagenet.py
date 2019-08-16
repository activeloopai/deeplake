import os
from PIL import Image
import numpy as np
import hub
from pathlib import Path
from pathos.threading import ThreadPool
import time

pool = ThreadPool(nodes=20)
#val_path = list(Path('./ILSVRC/Data/CLS-LOC/val').glob('*.JPEG'))
val_path = list(Path('./ILSVRC/Data/CLS-LOC/train').glob('**/*.JPEG'))
shape = (len(val_path), 500, 375, 3)
x = hub.array(shape, name='imagenet/test:latest', dtype='uint8')
print(x.shape)

index = 1
def upload_val(index):
    t1 = time.time()
    # Preprocess the image
    img = Image.open(val_path[index])
    img = img.resize((500,375), Image.ANTIALIAS)
    img = np.asarray(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    if img.shape[-1] == 4:
        img = img[...,:3]
    img = np.transpose(img, axes=(1,0,2))

    # Upload the image
    t2 = time.time()
    x[index] = np.expand_dims(img, 0)
    t3 = time.time()
    print("uploading {}/{}: downloded in {}s and uploaded in {}s ".format(index, len(val_path), t2-t1, t3-t2))

t1 = time.time()
list(pool.map(upload_val, list(range(len(val_path)))))
t2 = time.time()
print('uploaded {} images in {}s'.format(len(val_path), t2-t1))