"""CustomEvaluator：基于检索 ID 的轻量级评估器。

实现以下评估指标（全部为 [0.0, 1.0] 归一化值）：
- ``hit_rate``  : Hit Rate@K — 至少命中 1 个 golden_id 即为 1.0，否则为 0.0。
- ``mrr``       : Mean Reciprocal Rank — 第一个命中位置的倒数（1/rank）。
- ``precision`` : Precision@K — 检索结果中命中 golden_id 的比例。
- ``recall``    : Recall@K — golden_ids 中被检索命中的比例。

设计约定
--------
- 输入为字符串 ID 列表，不依赖任何外部模型或网络调用，纯内存计算。
- 空 golden_ids → 所有指标均为 0.0（无法评估时的保守策略）。
- 空 retrieved_ids → 所有指标均为 0.0（无检索结果）。
- 排名从 1 开始（rank-1 = retrieved_ids[0]）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalOutput,
    EvaluatorError,
)

try:
    from src.core.types import TraceContext
except ImportError:  # pragma: no cover
    TraceContext = None  # type: ignore[assignment,misc]

try:
    from src.core.settings import Settings
except ImportError:  # pragma: no cover
    Settings = None  # type: ignore[assignment,misc]


class CustomEvaluator(BaseEvaluator):
    """基于检索 ID 的轻量级评估器。

    计算无需外部模型的确定性指标：Hit Rate、MRR、Precision@K、Recall@K。

    Parameters
    ----------
    settings:
        全局 ``Settings`` 配置对象（当前实现不使用任何外部资源）。

    Examples
    --------
    >>> from src.libs.evaluator.base_evaluator import EvalInput
    >>> evaluator = CustomEvaluator()
    >>> inp = EvalInput(query="q", retrieved_ids=["c", "a", "b"], golden_ids=["a", "b"])
    >>> out = evaluator.evaluate(inp)
    >>> out.metrics["hit_rate"]
    1.0
    >>> out.metrics["mrr"]  # 'a' 在位置 2 → 1/2
    0.5
    """

    def __init__(self, settings: Optional["Settings"] = None) -> None:  # type: ignore[type-arg]
        self._settings = settings

    @property
    def backend(self) -> str:
        return "custom"

    def evaluate(
        self,
        eval_input: "EvalInput",
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
    ) -> "EvalOutput":
        """评估检索结果并返回各项指标。

        Parameters
        ----------
        eval_input:
            标准化评估输入，包含 query / retrieved_ids / golden_ids。
        trace:
            可选追踪上下文（CustomEvaluator 当前不打点，参数保留以满足接口约定）。

        Returns
        -------
        EvalOutput
            包含 hit_rate / mrr / precision / recall 四项指标的结果容器。

        Raises
        ------
        ValueError
            ``eval_input`` 为 ``None`` 时抛出。
        EvaluatorError
            内部计算发生不可恢复错误时抛出。
        """
        if eval_input is None:
            raise ValueError("eval_input 不能为 None")

        try:
            metrics = self._compute_metrics(
                retrieved_ids=eval_input.retrieved_ids,
                golden_ids=eval_input.golden_ids,
            )
        except Exception as exc:
            raise EvaluatorError(
                f"[{self.backend}] 指标计算失败: {exc}"
            ) from exc

        meta: Dict[str, Any] = {
            "retrieved_count": len(eval_input.retrieved_ids),
            "golden_count": len(eval_input.golden_ids),
        }

        return EvalOutput(backend=self.backend, metrics=metrics, meta=meta)

    # ------------------------------------------------------------------
    # 指标计算（内部，全部为确定性纯函数）
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        retrieved_ids: List[str],
        golden_ids: List[str],
    ) -> Dict[str, float]:
        """计算 hit_rate / mrr / precision / recall。

        Parameters
        ----------
        retrieved_ids:
            检索结果 ID 列表（顺序敏感，排名从 1 开始）。
        golden_ids:
            标准答案 ID 列表（顺序不敏感）。

        Returns
        -------
        Dict[str, float]
            各指标名称 → [0.0, 1.0] 浮点分值。
        """
        # 空输入 → 全零（保守策略）
        if not golden_ids or not retrieved_ids:
            return {
                "hit_rate": 0.0,
                "mrr": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        golden_set = set(golden_ids)

        # ── Hit Rate@K ────────────────────────────────────────────────
        # 只要检索结果中有任意一条命中 golden_ids 即为 1.0
        hit_rate = 1.0 if any(rid in golden_set for rid in retrieved_ids) else 0.0

        # ── MRR（Mean Reciprocal Rank）────────────────────────────────
        # 检索列表中第一个命中项的排名倒数；无命中 → 0.0
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in golden_set:
                mrr = 1.0 / rank
                break

        # ── Precision@K ───────────────────────────────────────────────
        # 检索结果中 golden 命中数 / 检索结果总数
        hits = sum(1 for rid in retrieved_ids if rid in golden_set)
        precision = hits / len(retrieved_ids)

        # ── Recall@K ──────────────────────────────────────────────────
        # golden_ids 中被检索到的比例
        recall = hits / len(golden_ids)

        return {
            "hit_rate": round(hit_rate, 6),
            "mrr": round(mrr, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
        }
