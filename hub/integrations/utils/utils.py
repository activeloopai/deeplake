from hub.core.dataset import Dataset

def is_hub_dataset(dataset):
    return isinstance(dataset, Dataset)
