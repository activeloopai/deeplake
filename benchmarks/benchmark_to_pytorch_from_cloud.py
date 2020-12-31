import torch
from tqdm import tqdm
import numpy as np

from hub import Dataset
from hub.schema import Tensor

def benchmark():
    schema = {
        "image": Tensor((1000, 1000, 4), dtype="float64")
    }
    arr = np.random.rand(1000, 1000, 4)
    ds = Dataset("s3://snark-test/superficial_dataset", mode="w", schema=schema, shape=(100,))
    for i in tqdm(range(len(ds))):
        ds["image", i] = arr
    ds.close()
    ds = Dataset("s3://snark-test/superficial_dataset")
    tds = ds.to_pytorch()
    dl = torch.utils.data.DataLoader(tds, batch_size=1, num_workers=8)
    for i, b in enumerate(tqdm(dl)):
        pass

if __name__ == "__main__":
    benchmark()