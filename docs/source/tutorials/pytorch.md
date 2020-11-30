# Pytorch Integration

In this tutorial we will retrieve our dataset from the local cache and integrate it with `Pytorch` for further use.

For a detailed guide on dataset generation and storage see [this tutorial](samples.md).

### Retreive from local cache

```python
import torch
from dataflow import hub_api
ds = hub_api.open("/cache_directory")
```

### Convert to `Pytorch`
```python
pytorch_ds = ds.to_pytorch(transform=lambda x: x)
```

### Create a `Dataloader`
```python
train_loader = torch.utils.data.DataLoader(
    pytorch_ds, batch_size=8, num_workers=1, collate_fn=pytorch_ds.collate_fn
)
```