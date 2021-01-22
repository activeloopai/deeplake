import numpy as np
from hub.schema.features import Tensor
import hub

"""Access credentials shown in this example belong to https://play.min.io:9000.
These credentials are open to public. Feel free to use this service for testing and development. Replace with your own MinIO keys in deployment."""

token = {
    "aws_access_key_id": "Q3AM3UQ867SPQQA43P2F",
    "aws_secret_access_key": "zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
    "endpoint_url": "https://play.min.io:9000",
    "region": "us-east-1",
}

schema = {"abc": Tensor((100, 100, 3))}
ds = hub.Dataset(
    "s3://mybucket/random_dataset", token=token, shape=(10,), schema=schema, mode="w"
)

for i in range(10):
    ds["abc", i] = np.ones((100, 100, 3))
ds.flush()
