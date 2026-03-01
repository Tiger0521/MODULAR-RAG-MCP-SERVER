"""Embedding 抽象基类与自定义异常。

设计原则
--------
- Pluggable       — 抽象接口 + ``EmbeddingFactory`` 按 provider 路由，上层无需感知具体实现。
- Config-Driven   — 具体实现构造函数接收 ``Settings``，provider 来自 settings.yaml。
- Observable      — ``embed()`` 接收可选 ``TraceContext``，供 F* 阶段打点使用。
- Graceful Error  — 统一使用 ``EmbeddingError`` 汇报可读错误，包含 provider 名称与原因。
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


class EmbeddingError(Exception):
    """Embedding 相关错误的统一基类。

    错误信息应包含 provider 名称与可读原因，便于排查配置或 API 问题。
    """


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseEmbedding(ABC):
    """Embedding 统一抽象接口。

    所有具体实现（OpenAI、Azure、Ollama、本地模型等）必须继承此类并实现
    ``provider`` 属性与 ``embed`` 方法。

    Interface Contract
    ------------------
    - ``provider`` 属性：返回与 ``settings.embedding.provider`` 匹配的小写字符串。
    - ``embed(texts, trace=None, **kwargs) -> list[list[float]]``：
      批量文本向量化，返回与输入等长的浮点向量列表。

    Raises
    ------
    EmbeddingError
        底层 API 调用失败或响应不符合预期时应统一转换并抛出 ``EmbeddingError``。
    ValueError
        ``texts`` 为空列表时应抛出 ``ValueError``（具体实现可自行选择是否严格校验）。
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """返回 provider 标识符（与 ``settings.embedding.provider`` 对应，小写）。"""

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> List[List[float]]:
        """批量文本向量化。

        Parameters
        ----------
        texts:
            待编码的文本列表。长度至少为 1；若为空列表，实现类可抛出
            ``ValueError`` 或返回空列表（推荐前者以快速暴露调用错误）。
        trace:
            可选追踪上下文，由上层 pipeline 传入；实现类可在此打点。
        **kwargs:
            透传给底层 SDK 的额外参数（如 ``batch_size``、``timeout``）。

        Returns
        -------
        list[list[float]]
            与 ``texts`` 等长的向量列表，每个向量为 ``list[float]``。
            向量维度由模型决定，调用方不应对维度做硬编码假设。

        Raises
        ------
        EmbeddingError
            底层 API 调用失败时抛出，错误信息包含 provider 名称与原始错误描述。
        ValueError
            ``texts`` 为空时抛出（推荐实现）。
        """

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def embed_one(self, text: str) -> List[float]:
        """对单条文本进行向量化，返回一维向量（不含 trace 支持）。

        适合脚本 / 测试使用；生产 pipeline 请使用 ``embed()``。
        """
        results = self.embed([text])
        return results[0]
