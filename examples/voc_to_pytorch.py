import torch
from hub import dataset

# Load data
ds = dataset.load("arenbeglaryan/vocsegmentation")



# Transform into pytorch
ds = ds.to_pytorch()

train_ds = torch.utils.data.DataLoader(
    ds, batch_size=2, num_workers=8, collate_fn=ds.collate_fn
)

# Iterate over the data
for batch in train_ds:
    print(batch["data"], batch["labels"])
    break






