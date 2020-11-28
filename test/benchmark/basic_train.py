import time
import torch
from helper import generate_dataset, report
import numpy as np
from PIL import Image
from pathlib import Path
import os
import tensorflow as tf


class PytorchDataset(torch.utils.data.Dataset):
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


def get_dataset_from_hub(samples=1, read_from_fs=False, pytorch=False):
    """
    Build dataset and transform to pytorch or tensorflow
    """
    ds = generate_dataset([(samples, 256, 256, 3), (samples, 1)])
    if read_from_fs:
        ds = ds.store("/tmp/training2")
    ds = ds.to_pytorch() if pytorch else ds.to_tensorflow()
    return ds


def TensorflowDataset(samples=100, load_image=False, image_path=""):
    def tf_gen(width=256):
        "Generates one sample of data"
        for i in range(samples):
            if load_image:
                folder = Path(__file__).parent
                path = os.path.join(folder, image_path)
                with open(path, "rb") as f:
                    img = Image.open(f)
                    inp = img.convert("RGB")
                    inp = np.array(inp)[:width, :width]
            else:
                inp = np.random.rand(width, width, 3)
                inp = (255 * inp).astype("uint8")
            objs = {"input": inp, "label": np.random.rand(1).astype("uint8")}
            yield objs

    ds = tf.data.Dataset.from_generator(
        tf_gen,
        output_types={
            "input": tf.dtypes.as_dtype("uint8"),
            "label": tf.dtypes.as_dtype("uint8"),
        },
        output_shapes={"input": [256, 256, 3], "label": [1]},
    )
    return ds


def dataset_loader(
    samples=1, read_from_fs=False, img_path="/tmp/test.png", pytorch=True
):
    """
    Returns tensorflow or pytorch dataset
    """
    inp = np.random.rand(256, 256, 3)
    inp = (255 * inp).astype("uint8")
    img = Image.fromarray(inp)
    img.save(img_path)

    Dataset = PytorchDataset if pytorch else TensorflowDataset
    ds = Dataset(samples=samples, load_image=read_from_fs, image_path=img_path)
    return ds


def empty_train_hub(samples=100, backend="hub:pytorch", read_from_fs=False):
    """
    Looping over empty space
    """
    if "hub" in backend:
        ds = get_dataset_from_hub(
            samples=samples,
            read_from_fs=read_from_fs,
            pytorch="pytorch" in backend,
        )
    else:
        ds = dataset_loader(
            samples=samples,
            read_from_fs=read_from_fs,
            pytorch="pytorch" in backend,
        )

    if "pytorch" in backend:
        ds = torch.utils.data.DataLoader(
            ds,
            batch_size=8,
            num_workers=8,
            collate_fn=ds.collate_fn if "collate_fn" in dir(ds) else None,
        )
    else:
        ds = ds.batch(16)

    t1 = time.time()
    for batch in ds:
        pass
    t2 = time.time()

    return {
        "name": f"{backend} loading from {'FS' if read_from_fs else 'RAM'}",
        # "samples": len(ds),
        "overall": t2 - t1,
        # "iterations": len(ds),
    }


if __name__ == "__main__":
    n_samples = 256
    params = [
        {"samples": n_samples, "backend": "pytorch", "read_from_fs": False},
        {"samples": n_samples, "backend": "pytorch", "read_from_fs": True},
        {"samples": n_samples, "backend": "hub:pytorch", "read_from_fs": False},
        {"samples": n_samples, "backend": "hub:pytorch", "read_from_fs": True},
        {"samples": n_samples, "backend": "tensorflow", "read_from_fs": False},
        {"samples": n_samples, "backend": "tensorflow", "read_from_fs": True},
        {"samples": n_samples, "backend": "hub:tensorflow", "read_from_fs": False},
        {"samples": n_samples, "backend": "hub:tensorflow", "read_from_fs": True},
    ]
    logs = [empty_train_hub(**args) for args in params]
    report(logs)
