"""Embedding 可插拔层公共 API。"""

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from src.libs.embedding.embedding_factory import EmbeddingFactory

__all__ = ["BaseEmbedding", "EmbeddingError", "EmbeddingFactory"]
