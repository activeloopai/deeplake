try:
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError(
        "Torch is not installed. Please install torch to use this feature."
    )

import deeplake


class TorchDataset(Dataset):

    def __init__(self, ds: deeplake.Dataset, transform=None):
        self.ds = ds
        self.transform = transform
        self.column_names = [col.name for col in ds.schema.columns]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        if self.transform:
            return self.transform(sample)
        else:
            out = {}
            for col in self.column_names:
                out[col] = sample[col]
            return out
