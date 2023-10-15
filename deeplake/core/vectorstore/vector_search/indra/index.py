from deeplake.core.distance_type import DistanceType
from deeplake.core.storage import azure, gcs, google_drive, local, lru_cache, memory
from deeplake.core.vectorstore import utils
from deeplake.constants import _INDEX_OPERATION_MAPPING
from enum import Enum


METRIC_TO_INDEX_METRIC = {
    "L2": "l2_norm",
    "L1": "l1_norm",
    "COS": "cosine_similarity",
}
class INDEX_OP_TYPE(Enum):
    NOOP = 0
    CREATE_INDEX = 1
    REMOVE_INDEX = 2
    REGENERATE_INDEX = 3
    INCREMENTAL_INDEX = 4



def parse_index_distance_metric_from_params(
    logger, distance_metric_index, distance_metric
):
    if distance_metric and distance_metric != distance_metric_index:
        logger.warning(
            f"The specified `distance_metric': `{distance_metric}` does not match the distance metric in the index: `{distance_metric_index}`."
            f"The search will be performed linearly the using specifed `distance_metric` and it will not use the index for ANN search. This is significantly slower compared to ANN search for >100k samples."
            "We reccommend you to specify the same `distance_metric` for both the index and the search, or leave the `distance_metric` parameter unspecified."
        )

        return distance_metric

    for key in METRIC_TO_INDEX_METRIC:
        if METRIC_TO_INDEX_METRIC[key] == distance_metric_index:
            return key

    raise ValueError(
        f"Invalid distance metric in the index: {distance_metric_index}. "
        f"Valid options are: {', '.join([e for e in list(METRIC_TO_INDEX_METRIC.keys())])}"
    )


def check_index_params(self):
    current_params = self.index_params
    existing_params = utils.get_embedding_tensor(self.dataset).get_vdb_indexes()[0]
    curr_distance_str = current_params.get("distance_metric", "COS")
    curr_distance = get_index_metric(curr_distance_str.upper())

    existing_distance = existing_params.get("distance", "COS")
    if curr_distance == existing_distance:
        current_additional_params_dict = current_params.get("additional_params", None)
        existing_additional_params_dict = existing_params.get("additional_params", None)
        if current_additional_params_dict == existing_additional_params_dict:
            return True

    return False

def index_operation_type(self, index_regeneration, index_delete=False):
    if not utils.index_used(self.exec_option):
       return INDEX_OP_TYPE.NOOP

    if index_delete:
        return INDEX_OP_TYPE.REMOVE_INDEX

    if not utils.index_exists(self.dataset):
        threshold = self.index_params.get("threshold", -1)
        below_threshold = threshold <= 0 or len(self.dataset) < threshold
        if not below_threshold:
            return INDEX_OP_TYPE.CREATE_INDEX
    else:
        if not index_regeneration and check_index_params(self):
            return INDEX_OP_TYPE.INCREMENTAL_INDEX
        else:
            return INDEX_OP_TYPE.REGENERATE_INDEX

    return INDEX_OP_TYPE.NOOP


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


# Routine to identify the index Operation.
def index_operation(self, dml_type, rowids, index_regeneration=False, index_delete=False):
    index_operation = index_operation_type(self, index_regeneration=index_regeneration,
                                           index_delete=index_delete)
    if index_operation == INDEX_OP_TYPE.NOOP:
        return

    emb_tensor = utils.get_embedding_tensor(self.dataset)
    if index_operation == INDEX_OP_TYPE.CREATE_INDEX:
        distance_str = self.index_params.get("distance_metric", "COS")
        additional_params_dict = self.index_params.get("additional_params", None)
        distance = get_index_metric(distance_str.upper())
        index_cache_cleanup(self.dataset)
        if additional_params_dict and len(additional_params_dict) > 0:
            param_dict = normalize_additional_params(additional_params_dict)
            emb_tensor.create_vdb_index(
                "hnsw_1", distance=distance, additional_params=param_dict
            )
        else:
            emb_tensor.create_vdb_index("hnsw_1", distance=distance)
    elif index_operation == INDEX_OP_TYPE.INCREMENTAL_INDEX:
        emb_tensor._incr_maintenance_vdb_indexes(rowids, dml_type)
    elif index_operation == INDEX_OP_TYPE.REGENERATE_INDEX:
        emb_tensor._regenerate_vdb_indexes()
    elif index_operation == INDEX_OP_TYPE.REMOVE_INDEX:
        vdb_indexes = emb_tensor.get_vdb_indexes()
        emb_tensor.delete_vdb_index(vdb_indexes["id"])
    else:
        raise Exception("Unknown index operation")


def validate_and_create_vector_index(dataset,
                                     index_params,
                                     regenerate_index=False,
                                     previous_dataset_len = 0):
    """
    Validate if the index is present in the dataset and create one if not present but required based on the specified index_params.
    Currently only supports 1 index per dataset.

    Returns: Distance metric for the index. If None, then no index is available.

    TODO: Update to support multiple indexes per dataset, only once the TQL parser also supports that
    """

    threshold = index_params.get("threshold", -1)

    below_threshold = threshold <= 0 or len(dataset) < threshold

    tensors = dataset.tensors

    # Check all tensors from the dataset.
    for _, tensor in tensors.items():
        is_embedding = utils.is_embedding_tensor(tensor)

        if is_embedding:
            vdb_indexes = tensor.get_vdb_indexes()

            if len(vdb_indexes) == 0 and not below_threshold:
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

                    return distance
                except ValueError as e:
                    raise e
            elif len(vdb_indexes) > 0:
                first_index = tensor.meta.vdb_indexes[0]
                distance = first_index["distance"]
                current_distance = index_params.get("distance_metric")
                if distance == METRIC_TO_INDEX_METRIC[current_distance.upper()]:
                    add_index = list(range(previous_dataset_len, len(dataset)))
                    tensor._incr_maintenance_vdb_indexes(add_index, _INDEX_OPERATION_MAPPING["ADD"])
                elif regenerate_index:
                    tensor._regenerate_vdb_indexes()

                return vdb_indexes[0]["distance"]

    return None
