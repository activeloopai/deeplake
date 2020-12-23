import hub
from hub.schema import ClassLabel, Image
from hub import Dataset 
import numpy as np


description = {
    "description":"testing data",
    "author":"worked",
    "license":"free",
    "citation":"null"
    
}

my_schema = {
    "image": Image((28, 28)),
    "label": ClassLabel(num_classes=10),
}

url = "./data/metatesting" #instead write your {username}/{dataset} to make it public

ds = Dataset(url, shape=(10,), schema=my_schema,mode="w",meta_information=description)
'''for i in range(len(ds)):
    ds["image", i] = np.ones((28, 28), dtype="uint8")
    ds["label", i] = 3

print(ds["image", 5].compute())
print(ds["label", 0:10].compute())'''
print(ds.meta)
ds.close()