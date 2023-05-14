from .indexer import Indexer
from .mutable_indexer import MutableIndexer

from typing import Callable, Dict, Tuple, Any, Optional, Iterable


class IndexFactory:
    factories: Dict[
        str,
        Tuple[
            Callable[[Dict[str, Any]], Indexer],
            Callable[[Dict[str, Any]], MutableIndexer],
        ],
    ] = dict()

    @staticmethod
    def register_index(
        type: str,
        mutable_factory: Callable[[Dict[str, Any]], MutableIndexer],
        factory: Optional[Callable[[Dict[str, Any]], Indexer]] = None,
    ):
        if factory is None:
            IndexFactory.factories[type] = (mutable_factory, mutable_factory)
        else:
            IndexFactory.factories[type] = (factory, mutable_factory)

    @staticmethod
    def supported_indices() -> Iterable[str]:
        return IndexFactory.factories.keys()

    @staticmethod
    def create_index(type: str, params: Dict[str, Any]) -> Indexer:
        return IndexFactory.factories[type][0](params)

    @staticmethod
    def create_mutable_index(type: str, params: Dict[str, Any]) -> MutableIndexer:
        return IndexFactory.factories[type][1](params)
