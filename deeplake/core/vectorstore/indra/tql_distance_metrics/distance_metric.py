def cosine_similarity(embeddings, query_embedding):
    return f"COSINE_SIMILARITY({embeddings}, {query_embedding})"


def l1_norm(embeddings, query_embedding):
    return f"L1_NORM({embeddings}-{query_embedding})"


def l2_norm(embeddings, query_embedding):
    return f"L2_NORM({embeddings}-{query_embedding})"


def linf_norm(embeddings, query_embedding):
    return f"LINF_NORM({embeddings}-{query_embedding})"


TQL_METRIC_TO_TQL_QUERY = {
    "l1": l1_norm,
    "l2": l2_norm,
    "cos": cosine_similarity,
    "max": linf_norm,
}


def get_tql_distance_metric(distance_metric, embeddings, query_embedding):
    metric_fn = TQL_METRIC_TO_TQL_QUERY[distance_metric]
    return metric_fn(embeddings, query_embedding)
