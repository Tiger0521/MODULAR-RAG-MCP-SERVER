"""Splitter 抽象基类与自定义异常。

设计原则
--------
- Pluggable       — 抽象接口 + ``SplitterFactory`` 按 provider 路由，上层无需感知具体实现。
- Config-Driven   — 具体实现构造函数接收 ``Settings``，provider 来自 settings.yaml。
- Observable      — ``split_text()`` 接收可选 ``TraceContext``，供 F* 阶段打点使用。
- Graceful Error  — 统一使用 ``SplitterError`` 汇报可读错误，包含 provider 名称与原因。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

# 延迟导入避免循环依赖；运行时才需要 TraceContext
try:
    from src.core.types import TraceContext
except ImportError:  # pragma: no cover
    TraceContext = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------


class SplitterError(Exception):
    """Splitter 相关错误的统一基类。

    错误信息应包含 provider 名称与可读原因，便于排查配置或实现问题。
    """


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseSplitter(ABC):
    """文本切分器统一抽象接口。

    所有具体实现（RecursiveSplitter、SemanticSplitter、FixedSplitter 等）
    必须继承此类并实现 ``provider`` 属性与 ``split_text`` 方法。

    Interface Contract
    ------------------
    - ``provider`` 属性：返回与 ``settings.splitter.provider`` 匹配的小写字符串。
    - ``split_text(text, trace=None, **kwargs) -> list[str]``：
      将输入文本切分为若干 chunk，返回非空字符串列表。

    Raises
    ------
    SplitterError
        切分过程中发生不可恢复错误时应统一转换并抛出 ``SplitterError``。
    ValueError
        ``text`` 为空字符串时应抛出 ``ValueError``（具体实现可自行选择是否严格校验）。
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """返回 provider 标识符（与 ``settings.splitter.provider`` 对应，小写）。"""

    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> List[str]:
        """将文本切分为 chunk 列表。

        Parameters
        ----------
        text:
            待切分的原始文本。若为空字符串，实现类可抛出 ``ValueError``
            或返回空列表（推荐前者以快速暴露调用错误）。
        trace:
            可选追踪上下文，由上层 pipeline 传入；实现类可在此打点。
        **kwargs:
            透传给具体实现的额外参数（如 ``chunk_size``、``chunk_overlap``）。

        Returns
        -------
        list[str]
            切分后的 chunk 列表，每个元素为非空字符串。
            若输入文本长度未超过阈值，可返回仅含原文本的单元素列表。

        Raises
        ------
        SplitterError
            切分逻辑发生不可恢复错误时抛出，错误信息包含 provider 名称。
        ValueError
            ``text`` 为空字符串时抛出（推荐实现）。
        """
