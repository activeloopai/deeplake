from hub.experimental.convert_to_hub3 import dataset_to_hub3
from hub.util.bugout_reporter import hub_reporter


@hub_reporter.record_call
def query(dataset, query_string: str):
    """Returns a sliced hub.Dataset with given query results.

    It allows to run SQL like queries on dataset and extract results. Currently supported keywords are the following:

    +-------------------------------------------+
    | SELECT                                    |
    +-------------------------------------------+
    | FROM                                      |
    +-------------------------------------------+
    | CONTAINS                                  |
    +-------------------------------------------+
    | ORDER BY                                  |
    +-------------------------------------------+
    | GROUP BY                                  |
    +-------------------------------------------+
    | LIMIT                                     |
    +-------------------------------------------+
    | OFFSET                                    |
    +-------------------------------------------+
    | RANDOM() -> for shuffling query results   |
    +-------------------------------------------+


    Args:
        dataset: hub.Dataset object on which the query needs to be run
        query_string (str): An SQL string adjusted with new functionalities to run on the given hub.Dataset object


    Returns:
        Dataset: A hub.Dataset object.

    Examples:

        Query from dataset all the samples with lables other than ``5``

        >>> import hub
        >>> from hub.experimental import query
        >>> ds = hub.load('hub://activeloop/fashion-mnist-train')
        >>> query_ds_train = query(ds_train, "select * where labels != 5")

        Query from dataset first appeard ``1000`` samples where the ``categories`` is ``car`` and ``1000`` samples where the ``categories`` is ``motorcycle``

        >>> ds_train = hub.load('hub://activeloop/coco-train')
        >>> query_ds_train = query(ds_train, "(select * where contains(categories, 'car') limit 1000) union (select * where contains(categories, 'motorcycle') limit 1000)")
    """
    ds = dataset_to_hub3(dataset)
    dsv = ds.query(query_string)
    indexes = dsv.indexes
    return dataset[indexes]
