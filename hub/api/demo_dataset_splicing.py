import numpy as np

import hub.api.dataset as dataset
from hub.features import Tensor

Dataset = dataset.Dataset

my_dtype = {
    "image": Tensor((10, 20, 30, 3), "uint8"),
    "label": {
        "a": Tensor((10, 12), "int32"),
        "b": Tensor((100, 400), "int64"),
    },
}


def test_dataset():
    
    ds = Dataset(
        "./data/hello_world", token=None, num_samples=10, mode="w+", dtype=my_dtype
    )
    # ds["label", 5, 50, 50:100,"a","bc"]

    #demo 1
    # for i in range(10):
    #     ds["image",i]=i*np.ones((10,20,30,3))
    # sub_ds=ds[3:9]
    # sub_sub_ds=sub_ds[2:4]
    # print(sub_sub_ds["image"].numpy())

    #demo 2
    # ds["label/a", 5, 5:, 2:8] = 7*np.ones((5,6))
    # print(ds["label",5,"a"].numpy())

    # demo 3
    # ds["label/a", 5, 5:, 2:8] = 7*np.ones((5,6))
    # sds=ds[4:6]
    # print(sds["label",1,"a"].numpy())

    #demo 4
    # ds["label","a"]=5*np.ones((10,10,12))
    # print(ds["label","a"].numpy().shape)

    # print(type(ds["/label/a"]))
    # print(ds["/label/a"].numpy())
    # print(ds[0])

    # assert ds["label/a", 5, 50, 50] == 8
    # ds["image", 5, 4, 100:200, 150:300, :] = np.ones((100, 150, 3), "uint8")
    # assert (
    #     ds["image", 5, 4, 100:200, 150:300, :] == np.ones((100, 150, 3), "uint8")
    # ).all()
    # # tuple
    # # print(ds["label/a"]) str
    # # print(ds[100]) int
    # # print(ds[100:200]) slice

    # for i in ds._tensors.keys():
    #     print(i)
    # print((ds._flat_tensors[1].path))


    # print(ds._tensors["/image"][:])
    # print(ds._tensors["/label"])


    # print(slice(10,1))
    # a=[]
    # for i in range(10):
    #     a.append(i)
    # print(a[11])
    # slice
    # print(slice().start)


if __name__ == "__main__":
    test_dataset()