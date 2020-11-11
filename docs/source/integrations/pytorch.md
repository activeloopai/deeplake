# PyTorch

Here is an example to transform the dataset into pytorch form.

```python
from hub import Dataset

# Create dataset
ds = Dataset(
    "username/pytorch_example",
    shape=(640,),
    mode="w",
    schema={
        "image": features.Tensor((512, 512), dtype="float"),
        "label": features.Tensor((512, 512), dtype="float"),
    },
)

# Load to pytorch
ds = ds.to_pytorch()
ds = torch.utils.data.DataLoader(
    ds,
    batch_size=8,
    num_workers=2,
)

# Iterate
for batch in ds:
    print(batch["image"], batch["label"])
```