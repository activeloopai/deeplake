import time
import torch
from hub.utils import generate_dataset, report
import numpy as np
from PIL import Image
from pathlib import Path
import os


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self, samples, width=256, load_image=True, image_path="results/Training.png"
    ):
        "Initialization"
        self.samples = samples
        self.width = width
        self.load_image = load_image
        self.image_path = image_path

    def __len__(self):
        "Denotes the total number of samples"
        return self.samples

    def __getitem__(self, index):
        "Generates one sample of data"
        if self.load_image:
            folder = Path(__file__).parent
            path = os.path.join(folder, self.image_path)
            with open(path, "rb") as f:
                img = Image.open(f)
                inp = img.convert("RGB")
                inp = np.array(inp)
        else:
            inp = np.random.rand(self.width, self.width, 3)
        return {"input": inp, "label": np.random.rand(1)}


def empty_train_hub(
    samples=100, use_hub=True, read_from_fs=False, name="Empty Training"
):
    """
    Looping over empty space
    """
    t1 = time.time()
    if use_hub:
        ds = generate_dataset([(samples, 256, 256, 3), (samples, 1)])
        t2 = time.time()
        if read_from_fs:
            ds = ds.store("/tmp/training")
        t3 = time.time()
        ds = ds.to_pytorch()
        t4 = time.time()
    else:
        ds = Dataset(samples=samples)
        t4 = t3 = t2 = time.time()

    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=8,
        num_workers=8,
        collate_fn=ds.collate_fn if "collate_fn" in dir(ds) else None,
    )
    t5 = time.time()

    for batch in train_loader:
        pass
        # your training loop here

    t6 = time.time()
    return {
        "name": name,
        "samples": len(ds),
        "overall": t6 - t5,
        "iterations": len(train_loader),
        "dataset_creation": t2 - t1,
        "writing to FS": t3 - t2,
        "to_pytorch": t4 - t3,
        "creating_loader": t5 - t4,
    }


if __name__ == "__main__":
    n_samples = 128
    r0 = empty_train_hub(n_samples, use_hub=False, name="Pytorch Dataset random")
    r1 = empty_train_hub(n_samples, name="Hub: Loading into RAM")
    r2 = empty_train_hub(n_samples, read_from_fs=True, name="Hub: Readig from FS")
    report([r0, r1, r2])
