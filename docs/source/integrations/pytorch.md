# PyTorch

## Dataset to PyTorch Dataset
Here is an example to transform the dataset into Pytorch form.

```python
import torch
from hub_v1 import dataset

# Create dataset
ds = Dataset(
    "username/pytorch_example",
    shape=(640,),
    mode="w",
    schema={
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    },
)

# Transform into Pytorch
ds = ds.to_pytorch(transform=None)
ds = torch.utils.data.DataLoader(
    ds,
    batch_size=8,
    num_workers=2,
)

# Iterate
for batch in ds:
    print(batch["image"], batch["label"])
```

## PyTorch Dataset to Dataset

You can also use `.from_pytorch()` to convert a PyTorch Dataset into Hub format.
```python
from torch.utils.data import Dataset

class TorchDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return 12

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        image = 5 * np.ones((50, 50))
        landmarks = 7 * np.ones((10, 10, 10))
        named = "testing text labels"
        sample = {
            "data": {"image": image, "landmarks": landmarks},
            "labels": {"named": named},
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

tds = TorchDataset()
ds = hub_v1.Dataset.from_pytorch(tds)
```