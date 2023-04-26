def l1_norm(embeddings, query_embedding, limit):
    return f"select * order by L1_NORM({embeddings}-ARRAY[{query_embedding}]) DESC LIMIT {limit}"


def l2_norm(embeddings, query_embedding, limit):
    return f"select * order by L2_NORM({embeddings}-ARRAY[{query_embedding}]) DESC LIMIT {limit}"


def cosine_similarity(embeddings, query_embedding, limit):
    return f"select * order by COSINE_SIMILARITY({embeddings}, ARRAY[{query_embedding}]) DESC LIMIT {limit}"


def linf_norm(embeddings, query_embedding, limit):
    return f"select * order by LINF_NORM({embeddings}-ARRAY[{query_embedding}]) DESC LIMIT {limit}"


def dot(embeddings, query_embedding, limit):
    return f"select * order by DOT({embeddings}, ARRAY[{query_embedding}]) DESC LIMIT {limit}"


TQL_METRIC_TO_TQL_QUERY = {
    "l1": l1_norm,
    "l2": l2_norm,
    "cos": cosine_similarity,
    "dot": dot,
    "max": linf_norm,
}
