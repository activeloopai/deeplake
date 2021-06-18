import pytest
from hub.constants import UNCOMPRESSED
from hub.tests.common import TENSOR_KEY
from hub import Dataset
from .test_api_with_compression import _populate_compressed_samples


@pytest.mark.full_benchmark
def test_compression(benchmark, local_ds: Dataset, cat_path, flower_path):
    images = local_ds.create_tensor(
        TENSOR_KEY, htype="image", sample_compression=UNCOMPRESSED
    )
    benchmark(_populate_compressed_samples, images, cat_path, flower_path, count=200)
