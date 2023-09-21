from deeplake.core.distance_type import DistanceType
from deeplake.core.storage import azure, gcs, google_drive, local, lru_cache, memory
from deeplake.core.vectorstore import utils


METRIC_TO_INDEX_METRIC = {
    "L2": "l2_norm",
    "COS": "cosine_similarity",
}


def get_index_distance_metric_from_params(logger, index_params, distance_metric):
    if distance_metric:
        logger.warning(
            f"Specifying `distance_metric` for a Vector Store with an index is not supported; `distance_metric` was specified as: `{distance_metric}`. "
            f"The search will be performed using the distance metric from index_params['distance_metric']: `{index_params['distance_metric']}`"
        )
    return index_params.get("distance_metric", "L2")


def get_index_metric(metric):
    if metric not in METRIC_TO_INDEX_METRIC:
        raise ValueError(
            f"Invalid distance metric: {metric} for index. "
            f"Valid options are: {', '.join([e for e in list(METRIC_TO_INDEX_METRIC.keys())])}"
        )
    return METRIC_TO_INDEX_METRIC[metric]


def normalize_additional_params(params: dict) -> dict:
    mapping = {"efconstruction": "efConstruction", "m": "M"}

    allowed_keys = ["efConstruction", "m"]

    # New dictionary to store the result with desired key format
    result_dict = {}

    for key, value in params.items():
        normalized_key = key.lower()

        # Check if the normalized key is one of the allowed keys
        if normalized_key not in mapping:
            raise ValueError(
                f"Unexpected key: {key} in additional_params"
                f" {allowed_keys} should be used instead."
            )

        # Check if the value is an integer
        if not isinstance(value, int):
            raise ValueError(
                f"Expected value for key {key} to be an integer, but got {type(value).__name__}"
            )

        # Populate the result dictionary with the proper format for the keys
        result_dict[mapping[normalized_key]] = value

    return result_dict


def check_vdb_indexes(dataset):
    tensors = dataset.tensors

    vdb_index_present = False
    for _, tensor in tensors.items():
        is_embedding = utils.is_embedding_tensor(tensor)
        has_vdb_indexes = hasattr(tensor.meta, "vdb_indexes")
        try:
            vdb_index_ids_present = len(tensor.meta.vdb_indexes) > 0
        except AttributeError:
            vdb_index_ids_present = False

        if is_embedding and has_vdb_indexes and vdb_index_ids_present:
            return True
    return False


def index_cache_cleanup(dataset):
    # Gdrive and In memory datasets are not supported for libdeeplake
    if dataset.path.startswith("gdrive://") or dataset.path.startswith("mem://"):
        return

    tensors = dataset.tensors
    for _, tensor in tensors.items():
        is_embedding = utils.is_embedding_tensor(tensor)
        if is_embedding:
            tensor.unload_index_cache()


def validate_and_create_vector_index(dataset, index_params, regenerate_index=False):
    threshold = index_params.get("threshold", -1)
    if threshold <= 0:
        return False
    elif len(dataset) < threshold:
        return False

    index_regen = False
    tensors = dataset.tensors
    # Check if regenerate_index is true.
    if regenerate_index:
        for _, tensor in tensors.items():
            is_embedding = utils.is_embedding_tensor(tensor)
            has_vdb_indexes = hasattr(tensor.meta, "vdb_indexes")
            try:
                vdb_index_ids_present = len(tensor.meta.vdb_indexes) > 0
            except AttributeError:
                vdb_index_ids_present = False

            if is_embedding and has_vdb_indexes and vdb_index_ids_present:
                tensor._regenerate_vdb_indexes()
                index_regen = True
        if index_regen:
            return

    # Check all tensors from the dataset.
    for _, tensor in tensors.items():
        is_embedding = utils.is_embedding_tensor(tensor)
        vdb_index_absent = len(tensor.meta.get_vdb_index_ids()) == 0
        if is_embedding and vdb_index_absent:
            try:
                distance_str = index_params.get("distance_metric", "L2")
                additional_params_dict = index_params.get("additional_params", None)
                distance = get_index_metric(distance_str.upper())
                if additional_params_dict and len(additional_params_dict) > 0:
                    param_dict = normalize_additional_params(additional_params_dict)
                    tensor.create_vdb_index(
                        "hnsw_1", distance=distance, additional_params=param_dict
                    )
                else:
                    tensor.create_vdb_index("hnsw_1", distance=distance)
            except ValueError as e:
                raise e

    return True
