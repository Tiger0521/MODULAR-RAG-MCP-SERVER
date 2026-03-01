"""VectorStore 抽象层公共导出。"""

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorStoreError,
    VectorRecord,
    QueryResult,
)
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

__all__ = [
    "BaseVectorStore",
    "VectorStoreError",
    "VectorRecord",
    "QueryResult",
    "VectorStoreFactory",
]
