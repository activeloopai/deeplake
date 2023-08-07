from deeplake.core.vector_index.distance_type import DistanceType
from deeplake.core.storage import azure, gcs, google_drive, local, lru_cache, memory


METRIC_TO_INDEX_METRIC = {
    "L2": "l2_norm",
    "COS": "cosine_similarity",
}


def get_index_distance_metric_from_params(logger, vector_index_params, distance_metric):
    if distance_metric:
        logger.warning(
            "specifying distance_metric for indexed dataset during the search "
            f"call is not supported. `distance_metric = {distance_metric}` "
            "specified during index creation will be used instead."
        )
    return vector_index_params.get('distance_metric', 'L2')


def get_index_metric(metric):
    if metric not in METRIC_TO_INDEX_METRIC:
        raise ValueError(
            f"Invalid distance metric: {metric} for index. "
            f"Valid options are: {', '.join([e for e in list(METRIC_TO_INDEX_METRIC.keys())])}"
        )
    return METRIC_TO_INDEX_METRIC[metric]


def validate_and_create_vector_index(dataset, vector_index_params):
    threshold = vector_index_params.get('threshold', 1000000)
    if threshold <= 0:
        return False
    elif len(dataset) < threshold:
        return False

    # Check whether the index is supported by the provider

    # Check all tensors from the dataset.
    tensors = dataset.tensors
    for _, tensor in tensors.items():
        is_embedding = tensor.htype == "embedding"
        vdb_index_absent = len(tensor.meta.get_vdb_index_ids()) == 0
        if is_embedding and vdb_index_absent:
            distance_str = vector_index_params.get('distance_metric', 'L2')
            distance = get_index_metric(distance_str.upper())
            tensor.create_vdb_index("hnsw_1", distance=distance)

    return True
