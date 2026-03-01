"""Evaluator 抽象基类与自定义异常。

设计原则
--------
- Pluggable       — 抽象接口 + ``EvaluatorFactory`` 按 backend 路由，上层无需感知具体实现。
- Config-Driven   — 具体实现构造函数接收 ``Settings``，backend 来自 settings.yaml。
- Observable      — ``evaluate()`` 接收可选 ``TraceContext``，供 F* 阶段打点使用。
- Graceful Error  — 统一使用 ``EvaluatorError`` 汇报可读错误，包含 backend 名称与原因。
- Deterministic   — 相同输入必须产生相同 metrics 输出（无随机性）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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


class EvaluatorError(Exception):
    """Evaluator 相关错误的统一基类。

    错误信息应包含 backend 名称与可读原因，便于排查配置或实现问题。
    """


# ---------------------------------------------------------------------------
# 评估输入/输出数据结构
# ---------------------------------------------------------------------------


class EvalInput:
    """评估输入数据容器。

    封装一次 RAG 评估所需的全部信息：查询、检索结果 ID 列表、标准答案 ID 列表。

    Parameters
    ----------
    query:
        查询字符串（用于上下文描述，部分指标可能利用语义相似度）。
    retrieved_ids:
        检索系统返回的文档/chunk ID 列表，**顺序敏感**（影响 MRR、NDCG 等排序指标）。
    golden_ids:
        人工标注的正确文档/chunk ID 列表（ground truth），顺序无关。
    extra:
        任意额外元数据，供扩展指标使用（如生成的答案文本、参考答案等）。
    """

    def __init__(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if retrieved_ids is None:
            raise ValueError("retrieved_ids 不能为 None")
        if golden_ids is None:
            raise ValueError("golden_ids 不能为 None")
        self.query = query
        self.retrieved_ids = list(retrieved_ids)
        self.golden_ids = list(golden_ids)
        self.extra: Dict[str, Any] = extra or {}

    def __repr__(self) -> str:
        return (
            f"EvalInput(query={self.query!r}, "
            f"retrieved_ids={self.retrieved_ids}, "
            f"golden_ids={self.golden_ids})"
        )


class EvalOutput:
    """评估输出数据容器。

    包含各指标名称到浮点分值的映射，以及 backend 标识和可选元数据。

    Parameters
    ----------
    backend:
        产生本次评估结果的 backend 名称（与 ``BaseEvaluator.backend`` 一致）。
    metrics:
        指标名称 → 浮点分值的字典（例如 ``{"hit_rate": 1.0, "mrr": 0.5}``）。
    meta:
        任意额外元数据（例如每个位置的命中情况等），供调试使用。
    """

    def __init__(
        self,
        backend: str,
        metrics: Dict[str, float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.backend = backend
        self.metrics: Dict[str, float] = dict(metrics)
        self.meta: Dict[str, Any] = meta or {}

    def __repr__(self) -> str:
        return f"EvalOutput(backend={self.backend!r}, metrics={self.metrics})"


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseEvaluator(ABC):
    """评估器统一抽象接口。

    所有具体实现（CustomEvaluator、RagasEvaluator 等）必须继承此类并实现
    ``backend`` 属性与 ``evaluate`` 方法。

    Interface Contract
    ------------------
    - ``backend`` 属性：返回与 ``settings.evaluator.backend`` 匹配的小写字符串。
    - ``evaluate(eval_input, trace=None) -> EvalOutput``：
      接收标准化输入，返回包含各指标浮点分值的 ``EvalOutput``；
      相同输入必须产生相同输出（确定性）。

    Raises
    ------
    EvaluatorError
        评估过程中发生不可恢复错误时应统一转换并抛出 ``EvaluatorError``。
    ValueError
        ``eval_input`` 为 ``None`` 时应抛出 ``ValueError``。
    """

    @property
    @abstractmethod
    def backend(self) -> str:
        """返回 backend 标识符（与 ``settings.evaluator.backend`` 对应，小写）。"""

    @abstractmethod
    def evaluate(
        self,
        eval_input: "EvalInput",
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
    ) -> "EvalOutput":
        """对单个查询的检索结果进行评估，返回各项指标。

        Parameters
        ----------
        eval_input:
            标准化评估输入，包含 query / retrieved_ids / golden_ids。
            不能为 ``None``；retrieved_ids 或 golden_ids 为空列表时各
            指标应返回 0.0（不视为错误）。
        trace:
            可选追踪上下文，由上层 pipeline 传入；实现类可在此打点耗时。

        Returns
        -------
        EvalOutput
            包含各指标浮点分值的结果容器；相同输入必须返回相同结果。

        Raises
        ------
        EvaluatorError
            评估逻辑发生不可恢复错误时抛出，错误信息包含 backend 名称。
        ValueError
            ``eval_input`` 为 ``None`` 时抛出。
        """
