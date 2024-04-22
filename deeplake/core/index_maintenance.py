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


def is_embedding_tensor(tensor):
    """Check if a tensor is an embedding tensor."""

    valid_names = ["embedding", "embeddings"]

    return (
        tensor.htype == "embedding"
        or tensor.meta.name in valid_names
        or tensor.key in valid_names
    )

def is_text_tensor(tensor):
    """Check if a tensor is a text tensor."""

    valid_names = ["text"]

    return (
        tensor.htype == "text"
        or tensor.meta.name in valid_names
        or tensor.key in valid_names
    )

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

    valid_names = ["text"]

    return (
        tensor.meta.name in valid_names and
        tensor.htype == "text" and
        tensor.key in valid_names
    )


def fetch_embedding_tensor(dataset):
    tensors = dataset.tensors
    for _, tensor in tensors.items():
        if validate_embedding_tensor(tensor):
            return tensor
    return None

def fetch_text_tensor(dataset):
    tensors = dataset.tensors
    for _, tensor in tensors.items():
        if validate_text_tensor(tensor):
            return tensor
    return None


def index_exists_for_embedding_tensor(dataset):
    """Check if the Index already exists."""
    emb_tensor = fetch_embedding_tensor(dataset)
    if emb_tensor is not None:
        vdb_indexes = emb_tensor.fetch_vdb_indexes()
        if len(vdb_indexes) == 0:
            return False
        else:
            return True
    else:
        return False

def index_exists_for_text_tensor(dataset):
    """Check if the Index already exists."""
    text_tensor = fetch_text_tensor(dataset)
    if text_tensor is not None:
        vdb_indexes = text_tensor.fetch_vdb_indexes()
        if len(vdb_indexes) == 0:
            return False
        else:
            return True
    else:
        return False


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
        current_additional_params_dict = current_params.get("additional_params", None)
        existing_additional_params_dict = existing_params.get("additional_params", None)
        if current_additional_params_dict == existing_additional_params_dict:
            return True

    return False


def index_operation_type_dataset(self, num_rows, changed_data_len):
    if not index_exists_for_embedding_tensor(self):
        if self.index_params is None:
            return INDEX_OP_TYPE.NOOP
        threshold = self.index_params.get("threshold", -1)
        below_threshold = threshold <= 0 or num_rows < threshold
        if not below_threshold:
            return INDEX_OP_TYPE.CREATE_INDEX

    if not check_vdb_indexes(self) or changed_data_len == 0:
        return INDEX_OP_TYPE.NOOP

    return INDEX_OP_TYPE.INCREMENTAL_INDEX


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
        is_embedding = is_embedding_tensor(tensor)
        has_vdb_indexes = hasattr(tensor.meta, "vdb_indexes")
        try:
            vdb_index_ids_present = len(tensor.meta.vdb_indexes) > 0
        except AttributeError:
            vdb_index_ids_present = False

        if is_embedding and has_vdb_indexes and vdb_index_ids_present:
            return True
    return False


def _incr_maintenance_vdb_indexes(tensor, indexes, index_operation):
    try:
        is_embedding = tensor.htype == "embedding"
        is_text = tensor.htype == "text"
        has_vdb_indexes = hasattr(tensor.meta, "vdb_indexes")
        try:
            vdb_index_ids_present = len(tensor.meta.vdb_indexes) > 0
        except AttributeError:
            vdb_index_ids_present = False

        if is_embedding or is_text and has_vdb_indexes and vdb_index_ids_present:
            for vdb_index in tensor.meta.vdb_indexes:
                tensor.update_vdb_index(
                    operation_kind=index_operation,
                    row_ids=indexes,
                )
    except Exception as e:
        raise Exception(f"An error occurred while regenerating VDB indexes: {e}")


# Routine to identify the index Operation.
def index_operation_vectorstore(self):
    if not index_used(self.exec_option):
        return None

    threshold = self.index_params.get("threshold", -1)
    below_threshold = threshold < 0 or len(self.dataset) < threshold
    if below_threshold:
        return None

    bm25 = self.index_params.get("bm25", False)
    print("BM25: ", bm25)
    if bm25:
        txt_tensor = fetch_text_tensor(self.dataset)

    emb_tensor = fetch_embedding_tensor(self.dataset)

    # TODO have to revisit it later.
    if index_exists_for_embedding_tensor(self.dataset) and check_index_params(self):
        return emb_tensor.get_vdb_indexes()[0]["distance"]

    if bm25 and index_exists_for_text_tensor(self.dataset):
        return txt_tensor.get_vdb_indexes()[0]

    # if not check_index_params(self):
    #     try:
    #         vdb_indexes = tensor.get_vdb_indexes()
    #         for vdb_index in vdb_indexes:
    #             tensor.delete_vdb_index(vdb_index["id"])
    #     except Exception as e:
    #         raise Exception(f"An error occurred while removing VDB indexes: {e}")


    if bm25:
        print("Creating BM25 index")
        txt_tensor.create_vdb_index("bm25")

    distance_str = self.index_params.get("distance_metric", "COS")
    additional_params_dict = self.index_params.get("additional_params", None)
    distance = get_index_metric(distance_str.upper())
    if additional_params_dict and len(additional_params_dict) > 0:
        param_dict = normalize_additional_params(additional_params_dict)
        print("Creating HNSW index")
        emb_tensor.create_vdb_index(
            "hnsw_1", distance=distance, additional_params=param_dict
        )
    else:
        print("Creating HNSW index")
        emb_tensor.create_vdb_index("hnsw_1", distance=distance)
    return distance


def index_operation_dataset(self, dml_type, rowids):
    if self.index_params is None:
        return

    bm25 = self.index_params.get("bm25", False)
    txt_tensor = None
    if bm25:
       txt_tensor = fetch_text_tensor(self)

    emb_tensor = fetch_embedding_tensor(self)
    if emb_tensor and txt_tensor is None:
        return

    num_rows = txt_tensor.chunk_engine.num_samples if txt_tensor is not None else emb_tensor.chunk_engine.num_samples

    index_operation_type = index_operation_type_dataset(
        self,
        num_rows,
        len(rowids),
    )

    if index_operation_type == INDEX_OP_TYPE.NOOP:
        return

    if (
        index_operation_type == INDEX_OP_TYPE.CREATE_INDEX
        or index_operation_type == INDEX_OP_TYPE.REGENERATE_INDEX
    ):
        if index_operation_type == INDEX_OP_TYPE.REGENERATE_INDEX:
            try:
                if txt_tensor is not None:
                    print("Regenerating BM25 index for text tensor")
                    vdb_indexes = txt_tensor.get_vdb_indexes()
                    for vdb_index in vdb_indexes:
                        txt_tensor.delete_vdb_index(vdb_index["id"])
                else:
                    vdb_indexes = emb_tensor.get_vdb_indexes()
                    for vdb_index in vdb_indexes:
                        emb_tensor.delete_vdb_index(vdb_index["id"])
            except Exception as e:
                raise Exception(
                    f"An error occurred while regenerating VDB indexes: {e}"
                )
        if txt_tensor is not None:
            print("Creating BM25 index")
            txt_tensor.create_vdb_index("bm25_1")

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
    elif index_operation_type == INDEX_OP_TYPE.INCREMENTAL_INDEX:
        if txt_tensor is not None:
            print("Incremental maintenance of BM25 index")
            _incr_maintenance_vdb_indexes(txt_tensor, rowids, dml_type)

        _incr_maintenance_vdb_indexes(emb_tensor, rowids, dml_type)
    else:
        raise Exception("Unknown index operation")
