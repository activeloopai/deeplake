from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.core.vectorstore.deep_memory import DeepMemory


class DeepMemoryVectorStore(VectorStore):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.deep_memory = DeepMemory(self.dataset)
