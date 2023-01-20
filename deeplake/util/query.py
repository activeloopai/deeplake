NONLINEAR_QUERY_OPPERATIONS = [
    "GROUP",
]


def is_linear_operation(query):
    if query is not None:
        for non_linear_query in NONLINEAR_QUERY_OPPERATIONS:
            if non_linear_query in query:
                return False
    return True
