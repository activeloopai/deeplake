# PyTorch

Here is an example to transform the dataset into Pytorch form.

```python
import torch
from hub import dataset

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