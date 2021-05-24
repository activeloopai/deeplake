def _to_pytorch(
    dataset,
    transform=None,
):
    try:
        import torch
    except ModuleNotFoundError:
        raise Exception
        # raise ModuleNotInstalledException("torch")

    global torch

    return TorchDataset(
        dataset,
        transform,
    )


class TorchDataset:
    def __init__(
        self,
        ds,
        transform=None,
    ):
        self.ds = ds
        self.transform = transform

    def _transform_data(self, data):
        return self.transform(data) if self.transform else data

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        d = {}
        for key in self.ds.tensors:
            item = self.ds[key][ind].numpy()
            if item.dtype == "uint16":
                item = item.astype("int32")
            elif item.dtype in ["uint32", "uint64"]:
                item = item.astype("int64")
            d[key] = item
        d = self._transform_data(d)
        return d

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
