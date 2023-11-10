from deeplake.core.vectorstore.vector_search import utils
from deeplake.core.vectorstore.vector_search.indra.vector_search import (
    vector_search as indra_vector_search,
)
from deeplake.core.vectorstore.vector_search.python.vector_search import (
    vector_search as python_vector_search,
)
from deeplake.core.vectorstore.vector_search.python.search_algorithm import (
    search as python_search_algorithm,
)
from deeplake.core.vectorstore.vector_search.indra.search_algorithm import (
    search as indra_search_algorithm,
)
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore

DeepLakeVectorStore = VectorStore
