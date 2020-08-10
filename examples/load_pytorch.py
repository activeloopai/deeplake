import torch
from hub import dataset

# Load data
ds = dataset.load("mnist/mnist")

# Transform into pytorch
ds = ds.to_pytorch()
ds = torch.utils.data.DataLoader(
    ds, batch_size=8, num_workers=8, collate_fn=ds.collate_fn
)

# Iterate over the data
for batch in ds:
    print(batch["data"], batch["labels"])
