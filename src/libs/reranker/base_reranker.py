"""Reranker 抽象基类、NoneReranker 默认实现与自定义异常。

设计原则
--------
- Pluggable       — 抽象接口 + ``RerankerFactory`` 按 backend 路由，上层无需感知具体实现。
- Config-Driven   — 具体实现构造函数接收 ``Settings``，backend 来自 settings.yaml。
- Observable      — ``rerank()`` 接收可选 ``TraceContext``，供 F* 阶段打点使用。
- Graceful Error  — 统一使用 ``RerankerError`` 汇报可读错误，包含 backend 名称与原因。
- Fallback Safe   — ``NoneReranker`` 作为默认回退，保持原排序不变，零副作用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

# 延迟导入避免循环依赖；运行时才需要 TraceContext
try:
    from src.core.types import TraceContext
except ImportError:  # pragma: no cover
    TraceContext = None  # type: ignore[assignment,misc]

try:
    from src.core.settings import Settings
except ImportError:  # pragma: no cover
    Settings = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------


class RerankerError(Exception):
    """Reranker 相关错误的统一基类。

    错误信息应包含 backend 名称与可读原因，便于排查配置或实现问题。
    """


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    """重排序器统一抽象接口。

    所有具体实现（NoneReranker、LLMReranker、CrossEncoderReranker 等）
    必须继承此类并实现 ``backend`` 属性与 ``rerank`` 方法。

    Interface Contract
    ------------------
    - ``backend`` 属性：返回与 ``settings.reranker.backend`` 匹配的小写字符串。
    - ``rerank(query, candidates, trace=None) -> list``：
      对候选列表按相关性重新排序并返回；候选元素类型由调用方约定。

    Raises
    ------
    RerankerError
        重排序过程中发生不可恢复错误时应统一转换并抛出 ``RerankerError``。
    ValueError
        ``candidates`` 为 ``None`` 时应抛出 ``ValueError``。
    """

    @property
    @abstractmethod
    def backend(self) -> str:
        """返回 backend 标识符（与 ``settings.reranker.backend`` 对应，小写）。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Any],
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
    ) -> List[Any]:
        """对候选集按 query 相关性重新排序。

        Parameters
        ----------
        query:
            查询字符串，用作重排序的参考依据。
        candidates:
            待重排序的候选列表，元素类型由调用方约定（通常为 QueryResult 或类似结构）。
            不能为 ``None``；空列表应直接返回空列表（不视为错误）。
        trace:
            可选追踪上下文，由上层 pipeline 传入；实现类可在此打点耗时。

        Returns
        -------
        list
            重排序后的候选列表，与输入元素类型一致。
            不得增加或减少元素（仅改变顺序），除非实现类明确说明截断行为
            （例如 top_n 截断）。

        Raises
        ------
        RerankerError
            重排序逻辑发生不可恢复错误时抛出，错误信息包含 backend 名称。
        ValueError
            ``candidates`` 为 ``None`` 时抛出。
        """


# ---------------------------------------------------------------------------
# NoneReranker：默认无操作回退
# ---------------------------------------------------------------------------


class NoneReranker(BaseReranker):
    """透传型 Reranker，保持候选列表原顺序不变。

    用作默认回退策略（backend="none"）。当不需要重排序或重排序服务不可用时，
    直接返回输入列表的浅拷贝，确保零副作用。

    Parameters
    ----------
    settings:
        全局 ``Settings`` 配置对象（当前实现不使用任何配置，保持接口一致）。
    """

    def __init__(self, settings: Optional["Settings"] = None) -> None:  # type: ignore[type-arg]
        self._settings = settings

    @property
    def backend(self) -> str:
        return "none"

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
    ) -> List[Any]:
        """直接返回候选列表的浅拷贝（排序不变）。

        Parameters
        ----------
        query:
            查询字符串（NoneReranker 不使用，保留以满足接口约定）。
        candidates:
            候选列表。
        trace:
            可选追踪上下文（NoneReranker 不打点）。

        Returns
        -------
        list
            与输入顺序完全相同的浅拷贝列表。

        Raises
        ------
        ValueError
            ``candidates`` 为 ``None`` 时抛出。
        """
        if candidates is None:
            raise ValueError("candidates 不能为 None")
        return list(candidates)
