from deeplake.core.distance_type import DistanceType
from deeplake.core.storage import azure, gcs, google_drive, local, lru_cache, memory
from enum import Enum


METRIC_TO_INDEX_METRIC = {
    "L2": "l2_norm",
    "L1": "l1_norm",
    "COS": "cosine_similarity",
}


class INDEX_OP_TYPE(Enum):
    NOOP = 0
    CREATE_INDEX = 1
    REGENERATE_INDEX = 2
    INCREMENTAL_INDEX = 3


def validate_embedding_tensor(tensor):
    """Check if a tensor is an embedding tensor."""

    valid_names = ["embedding"]

    return (
        tensor.htype == "embedding"
        or tensor.meta.name in valid_names
        or tensor.key in valid_names
    )


def validate_text_tensor(tensor):
    """Check if a tensor is an embedding tensor."""
    return tensor.htype == "text"


def fetch_embedding_tensor(dataset):
    tensors = dataset.tensors
    for _, tensor in tensors.items():
        if validate_embedding_tensor(tensor):
            return tensor
    return None


def index_exists_emb(emb_tensor):
    """Check if the Index already exists."""
    emb = validate_embedding_tensor(emb_tensor)
    if emb:
        vdb_indexes = emb_tensor.fetch_vdb_indexes()
        if len(vdb_indexes) == 0:
            return False
        else:
            return True
    return False


def index_exists_txt(txt_tensor):
    """Check if the Index already exists."""
    txt = validate_text_tensor(txt_tensor)
    if txt:
        vdb_indexes = txt_tensor.fetch_vdb_indexes()
        if len(vdb_indexes) == 0:
            return False
        else:
            return True
    return False


def index_partition_count(tensor):
    is_emb_tensor = validate_embedding_tensor(tensor)
    if is_emb_tensor:
        vdb_indexes = tensor.fetch_vdb_indexes()
        if len(vdb_indexes) == 0:
            return 1
        else:
            additional_params = vdb_indexes[0].get("additional_params", {})
            if additional_params is None:
                return 1
            return additional_params.get("partitions", 1)
    else:
        return 1


def index_used(exec_option):
    """Check if the index is used for the exec_option"""
    return exec_option in ("tensor_db", "compute_engine")


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
    emb_tensor = fetch_embedding_tensor(self.dataset)
    indexes = emb_tensor.get_vdb_indexes()
    if len(indexes) == 0:
        return False
    current_params = self.index_params
    existing_params = indexes[0]
    curr_distance_str = current_params.get("distance_metric", "COS")
    curr_distance = get_index_metric(curr_distance_str.upper())

    existing_distance = existing_params.get("distance", "COS")
    if curr_distance == existing_distance:
        current_additional_params_dict = current_params.get(
            "additional_params", {}
        ).copy()
        existing_additional_params_dict = existing_params.get(
            "additional_params", {}
        ).copy()

        # Remove the 'partitions' key from the copies of the dictionaries
        current_additional_params_dict.pop("partitions", None)
        existing_additional_params_dict.pop("partitions", None)

        if current_additional_params_dict == existing_additional_params_dict:
            return True

    return False


def index_operation_type_dataset(
    tensor, dataset, num_rows, changed_data_len, is_embedding=False
):
    if is_embedding:
        vdb_index_exists = index_exists_emb(tensor)
        if not vdb_index_exists:
            if dataset.index_params is None:
                return INDEX_OP_TYPE.NOOP
            if changed_data_len == 0:
                return INDEX_OP_TYPE.NOOP
            threshold = dataset.index_params.get("threshold", -1)
            below_threshold = threshold <= 0 or num_rows < threshold
            if not below_threshold:
                return INDEX_OP_TYPE.CREATE_INDEX

        return INDEX_OP_TYPE.INCREMENTAL_INDEX
    else:
        # for Text tensor i.e. inverted index,
        vdb_index_exists = index_exists_txt(tensor)
        if not vdb_index_exists or changed_data_len == 0:
            return INDEX_OP_TYPE.NOOP
        return INDEX_OP_TYPE.REGENERATE_INDEX


def get_index_metric(metric):
    if metric not in METRIC_TO_INDEX_METRIC:
        raise ValueError(
            f"Invalid distance metric: {metric} for index. "
            f"Valid options are: {', '.join([e for e in list(METRIC_TO_INDEX_METRIC.keys())])}"
        )
    return METRIC_TO_INDEX_METRIC[metric]


def normalize_additional_params(params: dict) -> dict:
    mapping = {
        "efconstruction": "efConstruction",
        "m": "M",
        "partitions": "partitions",
        "bloom": "bloom_filter_size",
        "bloom_size": "bloom_filter_size",
        "Segment_Size": "segment_size",
        "seg_size": "segment_size",
    }

    allowed_keys = [
        "efConstruction",
        "m",
        "partitions",
        "bloom_filter_size",
        "segment_size",
    ]

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


def check_embedding_vdb_indexes(tensor):
    is_embedding = validate_embedding_tensor(tensor)
    has_vdb_indexes = hasattr(tensor.meta, "vdb_indexes")
    try:
        vdb_index_ids_present = len(tensor.meta.vdb_indexes) > 0
    except AttributeError:
        vdb_index_ids_present = False

    if is_embedding and has_vdb_indexes and vdb_index_ids_present:
        return True
    return False


def _incr_maintenance_vdb_indexes(
    tensor, indexes, index_operation, is_partitioned=False
):
    try:
        is_embedding = tensor.htype == "embedding"
        has_vdb_indexes = hasattr(tensor.meta, "vdb_indexes")

        is_text = tensor.htype == "text"
        if is_text:
            raise Exception(
                "Inverted index does not support incremental index maintenance."
            )
        try:
            vdb_index_ids_present = len(tensor.meta.vdb_indexes) > 0
        except AttributeError:
            vdb_index_ids_present = False

        if (is_embedding or is_text) and has_vdb_indexes and vdb_index_ids_present:
            for vdb_index in tensor.meta.vdb_indexes:
                tensor.update_vdb_index(
                    operation_kind=index_operation,
                    row_ids=indexes,
                    is_partitioned=is_partitioned,
                )
    except Exception as e:
        raise Exception(f"An error occurred while regenerating VDB indexes: {e}")


def index_operation_vectorstore(self):
    if not index_used(self.exec_option):
        return None

    emb_tensor = fetch_embedding_tensor(self.dataset)
    if emb_tensor is None:
        return None

    if index_exists_emb(emb_tensor) and check_index_params(self):
        return emb_tensor.get_vdb_indexes()[0]["distance"]

    threshold = self.index_params.get("threshold", -1)
    below_threshold = threshold < 0 or len(self.dataset) < threshold
    if below_threshold:
        return None

    if not check_index_params(self):
        try:
            vdb_indexes = emb_tensor.get_vdb_indexes()
            for vdb_index in vdb_indexes:
                emb_tensor.delete_vdb_index(vdb_index["id"])
        except Exception as e:
            raise Exception(f"An error occurred while removing VDB indexes: {e}")
    distance_str = self.index_params.get("distance_metric", "COS")
    additional_params_dict = self.index_params.get("additional_params", None)
    distance = get_index_metric(distance_str.upper())
    if additional_params_dict and len(additional_params_dict) > 0:
        param_dict = normalize_additional_params(additional_params_dict)
        emb_tensor.create_vdb_index(
            "hnsw_1", distance=distance, additional_params=param_dict
        )
    else:
        emb_tensor.create_vdb_index("hnsw_1", distance=distance)
    return distance


def index_operation_dataset(self, dml_type, rowids):
    tensors = self.tensors
    for _, tensor in tensors.items():
        is_embedding_tensor = validate_embedding_tensor(tensor)
        is_text_tensor = validate_text_tensor(tensor)
        index_operation_type = INDEX_OP_TYPE.NOOP

        if is_embedding_tensor or is_text_tensor:
            index_operation_type = index_operation_type_dataset(
                tensor,
                self,
                tensor.chunk_engine.num_samples,
                len(rowids),
                is_embedding_tensor,
            )
        else:
            continue

        if index_operation_type == INDEX_OP_TYPE.NOOP:
            continue
        if (
            index_operation_type == INDEX_OP_TYPE.CREATE_INDEX
            or index_operation_type == INDEX_OP_TYPE.REGENERATE_INDEX
        ):
            saved_vdb_indexes = []
            if index_operation_type == INDEX_OP_TYPE.REGENERATE_INDEX:
                try:
                    vdb_indexes = tensor.get_vdb_indexes()
                    for vdb_index in vdb_indexes:
                        saved_vdb_indexes.append(vdb_index)
                        tensor.delete_vdb_index(vdb_index["id"])
                except Exception as e:
                    raise Exception(
                        f"An error occurred while regenerating VDB indexes: {e}"
                    )
            if is_embedding_tensor:
                distance_str = self.index_params.get("distance_metric", "COS")
                additional_params_dict = self.index_params.get(
                    "additional_params", None
                )
                distance = get_index_metric(distance_str.upper())
                if additional_params_dict and len(additional_params_dict) > 0:
                    param_dict = normalize_additional_params(additional_params_dict)
                    tensor.create_vdb_index(
                        "hnsw_1", distance=distance, additional_params=param_dict
                    )
                else:
                    tensor.create_vdb_index("hnsw_1", distance=distance)
            elif is_text_tensor:
                if len(saved_vdb_indexes) > 0:
                    for vdb_index in saved_vdb_indexes:
                        id = vdb_index["id"]
                        additional_params_dict = vdb_index.get(
                            "additional_params", None
                        )
                        if additional_params_dict and len(additional_params_dict) > 0:
                            param_dict = normalize_additional_params(
                                additional_params_dict
                            )
                            tensor.create_vdb_index(id, additional_params=param_dict)
                        else:
                            tensor.create_vdb_index(id)
            continue
        elif index_operation_type == INDEX_OP_TYPE.INCREMENTAL_INDEX:
            partition_count = index_partition_count(tensor)
            if partition_count > 1:
                _incr_maintenance_vdb_indexes(
                    tensor, rowids, dml_type, is_partitioned=True
                )
            else:
                _incr_maintenance_vdb_indexes(tensor, rowids, dml_type)
            continue
        else:
            raise Exception("Unknown index operation")
