from hub.experimental.convert_to_hub3 import dataset_to_hub3
from hub.util.bugout_reporter import hub_reporter


@hub_reporter.record_call
def query(dataset, query_string: str):
    """
    Query the dataset.
    """
    ds = dataset_to_hub3(dataset)
    dsv = ds.query(query_string)
    indexes = dsv.indexes  # TODO: enable this in indra
    return dataset[indexes]
