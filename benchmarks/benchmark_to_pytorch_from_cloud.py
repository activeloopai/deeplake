"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import torch
from tqdm import tqdm
import numpy as np

from hub import Dataset
from hub.schema import Tensor


def benchmark():
    schema = {"image": Tensor((256, 256, 3), dtype="uint8")}
    arr = (np.random.rand(256, 256, 3) * 100).astype("uint8")
    # ds = Dataset("s3://snark-test/superficial_dataset", mode="w", schema=schema, shape=(5000,))
    # for i in tqdm(range(len(ds))):
    #     ds["image", i] = arr
    # ds.close()
    ds = Dataset("s3://snark-test/superficial_dataset")
    tds = ds.to_pytorch()
    dl = torch.utils.data.DataLoader(tds, batch_size=32, num_workers=16)
    for i, b in enumerate(tqdm(dl)):
        pass


if __name__ == "__main__":
    benchmark()
