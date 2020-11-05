import torch
from hub import Dataset, features

# Create dataset
ds = Dataset(
    "./data/example/pytorch",
    shape=(64,),
    schema={
        "image": features.Tensor((512, 512), dtype="float"),
        "label": features.Tensor((512, 512), dtype="float"),
    },
)


# Transform into pytorch
ds = ds.to_pytorch()
ds = torch.utils.data.DataLoader(
    ds, batch_size=8, num_workers=0, collate_fn=ds.collate_fn
)

# Iterate over the data
for batch in ds:
    print(batch["image"], batch["label"])
