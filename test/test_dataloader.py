import hub
import torch
from torchvision import transforms
import torchvision.datasets as datasets

import torch
import numpy as np
from torchvision import transforms
import random

# from skimage.transform import rotate


def test_pytorch():
    # Create arrays
    datahub = hub.fs("./data/cache").connect()
    images = datahub.array(
        name="test/dataloaders/images3",
        shape=(100, 100, 100),
        chunk=(1, 100, 100),
        dtype="uint8",
    )
    labels = datahub.array(
        name="test/dataloaders/labels3", shape=(100, 1), chunk=(100, 1), dtype="uint8"
    )
    # Create dataset
    ds = datahub.dataset(
        name="test/loaders/dataset2", components={"images": images, "labels": labels}
    )
    # Transform to Pytorch
    train_dataset = ds.to_pytorch()
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    # Loop over attributes
    batch = next(iter(train_loader))
    assert batch["images"].shape == (32, 100, 100)
    assert batch["labels"].shape == (32, 1)


# TODO keras
# def test_keras():
#     print("not implemented")


def test_to_tensorflow():
    print("testing Tensorflow")
    conn = hub.fs("./data/cache").connect()
    ds = conn.open("test/loaders/dataset2")

    # Transform to Tensorflow
    train_dataset = ds.to_tensorflow()
    batch = next(iter(train_dataset.batch(batch_size=16)))
    assert batch["images"].shape == (16, 100, 100)
    # TODO create dataloader
