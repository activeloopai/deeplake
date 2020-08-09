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
        self,
        samples,
        width=256,
        load_image=True,
        image_path="results/Parallel150KB.png",
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
                inp = np.array(inp)[: self.width, : self.width]
        else:
            inp = np.random.rand(self.width, self.width, 3)
            inp = (255 * inp).astype("uint8")
        objs = {"input": inp, "label": np.random.rand(1).astype("uint8")}

        objs = {k: torch.tensor(v) for k, v in objs.items()}
        return objs

    def collate_fn(self, batch):
        batch = tuple(batch)
        keys = tuple(batch[0].keys())
        ans = {key: [item[key] for item in batch] for key in keys}

        for key in keys:
            ans[key] = torch.stack(ans[key], dim=0, out=None)
        return ans


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
        inp = np.random.rand(256, 256, 3)
        inp = (255 * inp).astype("uint8")
        im = Image.fromarray(inp)
        im.save("/tmp/test.png")
        ds = Dataset(
            samples=samples, load_image=read_from_fs, image_path="/tmp/test.png"
        )
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
        # "iterations": len(train_loader),
        # "dataset_creation": t2 - t1,
        # "writing to FS": t3 - t2,
        # "to_pytorch": t4 - t3,
    }


if __name__ == "__main__":
    n_samples = 256
    r0 = empty_train_hub(n_samples, use_hub=False, name="Pytorch Dataset From RAM")
    r1 = empty_train_hub(
        n_samples, use_hub=False, read_from_fs=True, name="Pytorch Dataset from FS"
    )
    r2 = empty_train_hub(n_samples, name="Hub: Loading from RAM")
    r3 = empty_train_hub(n_samples, read_from_fs=True, name="Hub: Reading from FS")
    report([r0, r1, r2, r3])
