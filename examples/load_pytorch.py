import torch
from hub import Dataset, schema


def main():
    # Create dataset
    ds = Dataset(
        "davitb/pytorch_example",
        shape=(640,),
        mode="w",
        schema={
            "image": schema.Tensor((512, 512), dtype="float"),
            "label": schema.Tensor((512, 512), dtype="float"),
        },
    )
    # ds["image"][:] = 1
    # ds["label"][:] = 2

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


# You need put inside a function, for pytorch multiprocesing to work
if __name__ == "__main__":
    main()
