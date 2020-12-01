from hub import Dataset, schema


def main():
    # Create dataset
    ds = Dataset(
        "./data/example/pytorch",
        shape=(64,),
        schema={
            "image": schema.Tensor((512, 512), dtype="float"),
            "label": schema.Tensor((512, 512), dtype="float"),
        },
    )

    # tansform into Tensorflow dataset
    ds = ds.to_tensorflow().batch(8)

    # Iterate over the data
    for batch in ds:
        print(batch["image"], batch["label"])


if __name__ == "__main__":
    main()
