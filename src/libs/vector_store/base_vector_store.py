"""VectorStore 抽象基类、数据契约与自定义异常。

设计原则
--------
- Pluggable       — 抽象接口 + ``VectorStoreFactory`` 按 provider 路由，上层无需感知具体实现。
- Config-Driven   — 具体实现构造函数接收 ``Settings``，provider 来自 settings.yaml。
- Observable      — 核心方法接收可选 ``TraceContext``，供 F* 阶段打点使用。
- Graceful Error  — 统一使用 ``VectorStoreError`` 汇报可读错误，包含 provider 名称与原因。
- Contract Clear  — ``VectorRecord`` 与 ``QueryResult`` 明确约束输入/输出 shape。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 延迟导入避免循环依赖；运行时才需要 TraceContext
try:
    from src.core.types import TraceContext
except ImportError:  # pragma: no cover
    TraceContext = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------


class VectorStoreError(Exception):
    """VectorStore 相关错误的统一基类。

    错误信息应包含 provider 名称与可读原因，便于排查配置或 DB 连接问题。
    """


# ---------------------------------------------------------------------------
# 数据契约（Data Contract）
# ---------------------------------------------------------------------------


@dataclass
class VectorRecord:
    """向量写入记录（upsert 输入契约）。

    Attributes
    ----------
    id:
        记录唯一标识符，具有幂等语义（相同 id 二次写入应覆盖而非追加）。
    vector:
        稠密向量（float 列表），维度由 Embedding 模型决定。
    text:
        原始文本内容，供检索结果返回给上层使用。
    metadata:
        任意结构化元数据（来源文件名、页码、chunk_index 等），用于过滤检索。
    """

    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("VectorRecord.id 不能为空字符串")
        if not self.vector:
            raise ValueError("VectorRecord.vector 不能为空列表")


@dataclass
class QueryResult:
    """向量检索结果（query 输出契约）。

    Attributes
    ----------
    id:
        命中记录的唯一标识符，与 ``VectorRecord.id`` 对应。
    score:
        相似度分数（越高越相似），范围由具体实现决定（通常 [0, 1] 或 [-1, 1]）。
    text:
        命中记录的原始文本。
    metadata:
        命中记录的元数据，与写入时 ``VectorRecord.metadata`` 一致。
    """

    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseVectorStore(ABC):
    """VectorStore 统一抽象接口。

    所有具体实现（ChromaStore、QdrantStore、PineconeStore 等）必须继承此类
    并实现 ``provider`` 属性与核心方法。

    Interface Contract
    ------------------
    - ``provider`` 属性：返回与 ``settings.vector_store.provider`` 匹配的小写字符串。
    - ``upsert(records, trace=None) -> int``：批量写入，返回写入条数。
    - ``query(vector, top_k, filters=None, trace=None) -> List[QueryResult]``：相似度检索。
    - ``delete(ids, trace=None) -> int``：按 ID 删除，返回实际删除条数。
    - ``delete_by_metadata(filter, trace=None) -> int``：按 metadata 条件批量删除。

    Raises
    ------
    VectorStoreError
        底层 DB 操作失败时应统一转换并抛出 ``VectorStoreError``。
    ValueError
        入参不符合契约要求（空列表、负数 top_k 等）时抛出。
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """返回 provider 标识符（与 ``settings.vector_store.provider`` 对应，小写）。"""

    @abstractmethod
    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> int:
        """批量写入（upsert）向量记录。

        Parameters
        ----------
        records:
            待写入的 ``VectorRecord`` 列表，长度至少为 1。
            若 id 已存在，实现应覆盖（幂等语义）。
        trace:
            可选追踪上下文。
        **kwargs:
            透传给底层 SDK 的额外参数。

        Returns
        -------
        int
            实际写入（新增 + 更新）的记录条数。

        Raises
        ------
        VectorStoreError
            写入失败时抛出，包含 provider 名称与原始错误描述。
        ValueError
            ``records`` 为空列表时抛出。
        """

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> List[QueryResult]:
        """相似度向量检索。

        Parameters
        ----------
        vector:
            查询向量（维度须与写入时一致）。
        top_k:
            返回结果数量上限，必须 >= 1。
        filters:
            可选 metadata 过滤条件（key-value 精确匹配），``None`` 表示不过滤。
        trace:
            可选追踪上下文。
        **kwargs:
            透传给底层 SDK 的额外参数。

        Returns
        -------
        List[QueryResult]
            按相似度降序排列的结果列表，长度 <= top_k。

        Raises
        ------
        VectorStoreError
            检索失败时抛出。
        ValueError
            ``vector`` 为空或 ``top_k < 1`` 时抛出。
        """

    @abstractmethod
    def delete(
        self,
        ids: List[str],
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> int:
        """按 ID 批量删除记录。

        Parameters
        ----------
        ids:
            待删除记录的 ID 列表，长度至少为 1。
        trace:
            可选追踪上下文。

        Returns
        -------
        int
            实际删除的记录条数（不存在的 ID 不计入）。

        Raises
        ------
        VectorStoreError
            删除操作失败时抛出。
        ValueError
            ``ids`` 为空列表时抛出。
        """

    @abstractmethod
    def delete_by_metadata(
        self,
        filter: Dict[str, Any],
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> int:
        """按 metadata 条件批量删除记录。

        Parameters
        ----------
        filter:
            metadata 过滤条件（key-value 精确匹配），不能为空 dict
            （防止误删全库）。
        trace:
            可选追踪上下文。

        Returns
        -------
        int
            实际删除的记录条数。

        Raises
        ------
        VectorStoreError
            删除操作失败时抛出。
        ValueError
            ``filter`` 为空 dict 时抛出。
        """
