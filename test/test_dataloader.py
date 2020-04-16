import hub
import torch
from torchvision import transforms
import torchvision.datasets as datasets

import torch
import numpy as np
from torchvision import transforms
import random
from skimage.transform import rotate

def test_pytorch():
    print('testing pytorch')

    # Create arrays
    images = hub.array((100000, 100, 100),
                       name='test/dataloaders:images', dtype='uint8')
    labels = hub.array(
        (100000, 1), name='test/dataloaders:labels', dtype='uint8')

    # Create dataset
    ds = hub.dataset({
        'images': images,
        'labels': labels
    }, name='test/loaders:dataset')

    # Transform to Pytorch
    train_dataset = ds.to_pytorch()

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, num_workers=1,
        pin_memory=True
    )

    # Loop over attributes
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape, labels.shape)
        assert (images.shape == (32, 100, 100))
        assert (labels.shape == (32, 1))
        break

    print('pass')


def test_pytorch_new():
    print('testing pytorch new')

    # Create arrays
    conn = hub.fs('./data/cache').connect()
    images = conn.array('test/test1/image2', (1000, 100, 100, 3), chunk=(100, 100, 100, 3), dtype='uint8')
    labels = conn.array('test/test1/label2', (1000, 1) , chunk=(100, 1), dtype='uint8')
    masks = conn.array('test/test1/mask2', (1000, 100, 100) , chunk=(100, 100, 100), dtype='uint8')
    
    

    # Create dataset
    ds = conn.dataset(name='test/test1/loaders2', components={
        'image': images,
        'label': labels,
        'mask': masks
    })

    # Transform to Pytorch
    train_dataset = ds.to_pytorch(transform = ToTensor())

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, num_workers=4,
        pin_memory=True
    )

    # Loop over attributes
    for i, batch in enumerate(train_loader):
        for key, item in batch.items():
            if key == "image":
                print(key)
                print(item.shape)
                # assert (item.shape == (32, 100, 100, 3))
            if key == "label":
                print(key)
                print(item.shape)
                pass
                # assert (item.shape == (32, 1))
            if key == "mask":
                print(key)
                print(item.shape)
                pass
                # assert (item.shape == (32, 100, 100))
            break

    print('pass')


def test_keras():
    print('testing keras')
    ...
    print('not implemented')


def test_tensorflow():
    print('testing Tensorflow')
    ...
    print('Not implemented')


if __name__ == "__main__":
    # test_pytorch()
    test_pytorch_new()
    exit()
    # test_keras()
    # test_tensorflow()
