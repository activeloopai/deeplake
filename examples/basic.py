from hub import Dataset, schema
import numpy as np


def main():
    # Tag is set {Username}/{Dataset}
    tag = "davitb/basic11"

    # Create dataset
    ds = Dataset(
        tag,
        shape=(4,),
        schema={
            "image": schema.Tensor((512, 512), dtype="float"),
            "label": schema.Tensor((512, 512), dtype="float"),
        },
        mode="w+",
    )

    # Upload Data
    ds["image"][:] = np.ones((4, 512, 512))
    ds["label"][:] = np.ones((4, 512, 512))
    ds.flush()

    # Load the data
    ds = Dataset(tag)
    print(ds["image"][0].compute())


if __name__ == "__main__":
    main()
