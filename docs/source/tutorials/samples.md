# Dataset Generation

In this example we will generate a dataset of endrows.

### Setup
Initialize Hub API
```python
from dataflow import sampler, ingestor, hub_api
hub_api.init()
```

### Flight-codes
Query for endrow flight-codes.
```python
flight_codes = ingestor.intelinair.flight_codes_query(
    [ingestor.intelinair.UIUC_ENDROW], range_=(0, 4)
)
```
### Fields
Generate a dataset of fields
```python
ingest_obj = ingestor.Intelinair(
    flight_codes=flight_codes,
    alert_types=[ingestor.intelinair.UIUC_ENDROW],
    url=_intelinair_bucket_path,
    creds=_intelinair_creds_path,
    channels=["nir","red","green"],
)
fields_ds = ingest_obj()
```
### Samples
Generate a dataset of samples
```python
sampler_obj = sampler.PolygonSampler((3, 512, 1024), ds)
ds = sampler_obj()
```

```eval_rst
.. note::

    Please note that no computations were made up to this point. The dataset is lazily loaded and will not be generated until instructed so.
```
### Local Cache
We can now create a local cache of the data. The function will first compute the dataset and then store it locally.
```python
hub_api.cache(train, name="/cache_directory", clear=True)
```