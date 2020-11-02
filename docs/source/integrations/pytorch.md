# PyTorch

Here is an example to transform the dataset into Pytorch form.

```python
import torch
from hub import dataset

# Load data
ds = dataset.load("mnist/mnist")

# Transform into Pytorch
ds = ds.to_pytorch(transform=None)
ds = torch.utils.data.DataLoader(
    ds, batch_size=8, num_workers=8, collate_fn=ds.collate_fn
)

# Iterate over the data
for batch in ds:
    print(batch["data"], batch["labels"])
```
Please make sure that `collate_fn` is provided from the dataset `ds.collate_fn` to stack tensors together since they are in dictionary form