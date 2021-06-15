import hub_v1
import torch


def main():
    ds = hub_v1.Dataset("eurosat/eurosat-rgb")

    # 26000 samples in dataset, accessing values
    print(ds["image"][10].numpy())
    print(
        ds["label", 15].numpy()
    )  # alternate way to access, by specifying both key and sample number at once
    print(ds["filename", 20:22].numpy())  # accessing multiple elements at once

    # Splitting into train and test sets
    train_ds = ds[:13000]
    test_ds = ds[13000:]

    # Using hub with tensorflow
    train_tf_ds = train_ds.to_tensorflow().batch(2)

    for batch in train_tf_ds:
        print(batch["label"], batch["filename"], batch["image"])
        break

    test_tf_ds = test_ds.to_tensorflow().batch(2)

    for batch in test_tf_ds:
        print(batch["label"], batch["filename"], batch["image"])
        break

    # Using hub with pytorch
    train_pt_ds = train_ds.to_pytorch()
    train_loader = torch.utils.data.DataLoader(train_pt_ds, batch_size=2)

    for batch in train_loader:
        print(
            batch["label"], batch["image"]
        )  # pytorch tensors don't support text labels such as filename
        break

    test_pt_ds = test_ds.to_pytorch()
    test_loader = torch.utils.data.DataLoader(test_pt_ds, batch_size=2)
    for batch in test_loader:
        print(
            batch["label"], batch["image"]
        )  # pytorch tensors don't support text labels such as filename
        break


if __name__ == "__main__":
    main()
