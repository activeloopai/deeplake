from hub import dataset


# Load data
ds = dataset.load("arenbeglaryan/vocsegmentation")

ds = ds.to_tensorflow().batch(8)

# Iterate over the data
for batch in ds:
    print(batch["data"], batch["labels"])

