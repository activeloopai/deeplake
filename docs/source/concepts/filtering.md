# Dataset Filtering

Using Hub you can filter your dataset to get a DatasetView that only has the items that you're interested in.
Filtering can be applied both to a Dataset or to a DatasetView (obtained by slicing or filtering a Dataset)

## Filtering using a function
Using filter, you can pass in a function that is applied element by element to the dataset. Only those elements for which the function returns True stay in the newly created DatasetView.

Example:-

```python
my_schema = {
    "img": Tensor((100, 100)),
    "name": Text((None,), max_shape=(10,))
}
ds = hub.Dataset("./data/filtering_example", shape=(20,), schema=my_schema)
for i in range(10):  # assigning some values to the dataset
    ds["img", i] = np.ones((100, 100))
    ds["name", i] = "abc" + str(i) if i % 2 == 0 else "def" + str(i)

def my_filter(sample):
    return sample["name"].compute().startswith("abc") and (sample["img"].compute() == np.ones((100, 100))).all()
ds2 = ds.filter(my_filter)

# alternatively, we can also use a lambda function to achieve the same results
ds3 = ds.filter(
    lambda x: x["name"].compute().startswith("abc")
    and (x["img"].compute() == np.ones((100, 100))).all()
)
```

## API
```eval_rst
.. autofunction:: hub.api.dataset.Dataset.filter
.. autofunction:: hub.api.datasetview.DatasetView.filter
```



