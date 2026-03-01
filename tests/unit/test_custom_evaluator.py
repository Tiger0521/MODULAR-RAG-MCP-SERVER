"""Unit tests for B6: Evaluator 抽象接口与工厂

验收标准
--------
- 输入 query + retrieved_ids + golden_ids 能输出稳定 metrics。
- hit_rate: 至少命中 1 个 golden_id 即为 1.0，否则为 0.0。
- mrr: 第一个命中位置的倒数（1/rank），无命中时为 0.0。
- precision: 检索结果中命中比例。
- recall: golden_ids 中被检索到的比例。
- 空 retrieved_ids / golden_ids → 全零安全返回。
- eval_input=None → ValueError。
- EvaluatorFactory.register() + create() 路由正确。
- 未知 backend 时 EvaluatorError 包含 backend 名称与已注册列表。
- evaluate() 接受可选 trace 参数且不影响结果。
- 多 backend 注册后按 settings.evaluator.backend 正确路由。
- 同输入 → 同输出（确定性验证）。
"""

import pytest
from typing import Dict, List, Optional

from src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvalInput,
    EvalOutput,
    EvaluatorError,
)
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.core.settings import Settings
from src.core.types import TraceContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(backend: str = "custom") -> Settings:
    """快速构造 Settings，仅覆盖 evaluator.backend。"""
    s = Settings()
    s.evaluator.backend = backend
    return s


def make_eval_input(
    retrieved_ids: List[str],
    golden_ids: List[str],
    query: str = "test query",
) -> EvalInput:
    return EvalInput(query=query, retrieved_ids=retrieved_ids, golden_ids=golden_ids)


# ---------------------------------------------------------------------------
# Fixture：每个测试结束后清理注册表
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试结束后清理 EvaluatorFactory 注册表，防止测试间污染。"""
    yield
    EvaluatorFactory._registry.clear()


# ---------------------------------------------------------------------------
# 1. EvalInput 数据类测试
# ---------------------------------------------------------------------------


class TestEvalInput:
    def test_basic_construction(self):
        inp = EvalInput(
            query="q",
            retrieved_ids=["a", "b", "c"],
            golden_ids=["a", "c"],
        )
        assert inp.query == "q"
        assert inp.retrieved_ids == ["a", "b", "c"]
        assert inp.golden_ids == ["a", "c"]
        assert inp.extra == {}

    def test_extra_metadata(self):
        inp = EvalInput(
            query="q",
            retrieved_ids=["a"],
            golden_ids=["a"],
            extra={"answer": "Paris"},
        )
        assert inp.extra["answer"] == "Paris"

    def test_retrieved_ids_none_raises(self):
        with pytest.raises(ValueError, match="retrieved_ids"):
            EvalInput(query="q", retrieved_ids=None, golden_ids=["a"])  # type: ignore[arg-type]

    def test_golden_ids_none_raises(self):
        with pytest.raises(ValueError, match="golden_ids"):
            EvalInput(query="q", retrieved_ids=["a"], golden_ids=None)  # type: ignore[arg-type]

    def test_input_lists_are_copies(self):
        """修改原始列表不影响 EvalInput 内部状态。"""
        original = ["a", "b"]
        inp = EvalInput(query="q", retrieved_ids=original, golden_ids=["c"])
        original.append("x")
        assert inp.retrieved_ids == ["a", "b"]

    def test_repr_contains_query(self):
        inp = EvalInput(query="hello", retrieved_ids=["a"], golden_ids=["a"])
        assert "hello" in repr(inp)


# ---------------------------------------------------------------------------
# 2. EvalOutput 数据类测试
# ---------------------------------------------------------------------------


class TestEvalOutput:
    def test_basic_construction(self):
        out = EvalOutput(backend="custom", metrics={"hit_rate": 1.0, "mrr": 0.5})
        assert out.backend == "custom"
        assert out.metrics["hit_rate"] == 1.0
        assert out.metrics["mrr"] == 0.5
        assert out.meta == {}

    def test_with_meta(self):
        out = EvalOutput(
            backend="custom",
            metrics={"hit_rate": 0.0},
            meta={"retrieved_count": 3},
        )
        assert out.meta["retrieved_count"] == 3

    def test_metrics_are_copy(self):
        m = {"hit_rate": 1.0}
        out = EvalOutput(backend="b", metrics=m)
        m["extra"] = 99.0
        assert "extra" not in out.metrics

    def test_repr_contains_backend(self):
        out = EvalOutput(backend="custom", metrics={})
        assert "custom" in repr(out)


# ---------------------------------------------------------------------------
# 3. CustomEvaluator — 核心指标测试
# ---------------------------------------------------------------------------


class TestCustomEvaluatorMetrics:
    """验证每个指标的计算逻辑。"""

    def setup_method(self):
        self.ev = CustomEvaluator()

    # ── hit_rate ────────────────────────────────────────────────────────

    def test_hit_rate_full_match(self):
        inp = make_eval_input(["a", "b", "c"], ["a", "b"])
        out = self.ev.evaluate(inp)
        assert out.metrics["hit_rate"] == 1.0

    def test_hit_rate_partial_match(self):
        inp = make_eval_input(["x", "a", "y"], ["a"])
        out = self.ev.evaluate(inp)
        assert out.metrics["hit_rate"] == 1.0

    def test_hit_rate_no_match(self):
        inp = make_eval_input(["x", "y"], ["a", "b"])
        out = self.ev.evaluate(inp)
        assert out.metrics["hit_rate"] == 0.0

    def test_hit_rate_empty_retrieved(self):
        inp = make_eval_input([], ["a"])
        out = self.ev.evaluate(inp)
        assert out.metrics["hit_rate"] == 0.0

    def test_hit_rate_empty_golden(self):
        inp = make_eval_input(["a"], [])
        out = self.ev.evaluate(inp)
        assert out.metrics["hit_rate"] == 0.0

    # ── MRR ─────────────────────────────────────────────────────────────

    def test_mrr_first_position(self):
        """命中 rank-1 → MRR = 1.0"""
        inp = make_eval_input(["a", "b", "c"], ["a"])
        out = self.ev.evaluate(inp)
        assert out.metrics["mrr"] == pytest.approx(1.0)

    def test_mrr_second_position(self):
        """命中 rank-2 → MRR = 0.5"""
        inp = make_eval_input(["x", "a", "c"], ["a"])
        out = self.ev.evaluate(inp)
        assert out.metrics["mrr"] == pytest.approx(0.5)

    def test_mrr_third_position(self):
        """命中 rank-3 → MRR ≈ 0.333"""
        inp = make_eval_input(["x", "y", "a"], ["a"])
        out = self.ev.evaluate(inp)
        assert out.metrics["mrr"] == pytest.approx(1 / 3, abs=1e-5)

    def test_mrr_first_hit_wins(self):
        """多个命中时取第一个命中位置。"""
        inp = make_eval_input(["x", "a", "b"], ["a", "b"])
        out = self.ev.evaluate(inp)
        assert out.metrics["mrr"] == pytest.approx(0.5)  # 'a' 在位置 2

    def test_mrr_no_match(self):
        inp = make_eval_input(["x", "y"], ["a"])
        out = self.ev.evaluate(inp)
        assert out.metrics["mrr"] == 0.0

    def test_mrr_empty_retrieved(self):
        inp = make_eval_input([], ["a"])
        out = self.ev.evaluate(inp)
        assert out.metrics["mrr"] == 0.0

    # ── Precision ────────────────────────────────────────────────────────

    def test_precision_all_hit(self):
        """全部命中 → precision = 1.0"""
        inp = make_eval_input(["a", "b"], ["a", "b"])
        out = self.ev.evaluate(inp)
        assert out.metrics["precision"] == pytest.approx(1.0)

    def test_precision_half_hit(self):
        """检索 2 条，命中 1 条 → precision = 0.5"""
        inp = make_eval_input(["a", "x"], ["a"])
        out = self.ev.evaluate(inp)
        assert out.metrics["precision"] == pytest.approx(0.5)

    def test_precision_no_hit(self):
        inp = make_eval_input(["x", "y"], ["a"])
        out = self.ev.evaluate(inp)
        assert out.metrics["precision"] == 0.0

    def test_precision_empty_retrieved(self):
        inp = make_eval_input([], ["a"])
        out = self.ev.evaluate(inp)
        assert out.metrics["precision"] == 0.0

    # ── Recall ───────────────────────────────────────────────────────────

    def test_recall_all_golden_hit(self):
        """golden 全部被检索到 → recall = 1.0"""
        inp = make_eval_input(["a", "b", "c"], ["a", "b"])
        out = self.ev.evaluate(inp)
        assert out.metrics["recall"] == pytest.approx(1.0)

    def test_recall_half_golden_hit(self):
        """golden 2 条命中 1 条 → recall = 0.5"""
        inp = make_eval_input(["a", "x"], ["a", "b"])
        out = self.ev.evaluate(inp)
        assert out.metrics["recall"] == pytest.approx(0.5)

    def test_recall_no_hit(self):
        inp = make_eval_input(["x"], ["a", "b"])
        out = self.ev.evaluate(inp)
        assert out.metrics["recall"] == 0.0

    def test_recall_empty_golden(self):
        inp = make_eval_input(["a"], [])
        out = self.ev.evaluate(inp)
        assert out.metrics["recall"] == 0.0

    # ── 综合场景 ──────────────────────────────────────────────────────────

    def test_combined_typical_scenario(self):
        """
        retrieved=[c, a, b], golden=[a, b]
        - hit_rate=1.0 (a 在 rank-2)
        - mrr=0.5 (a 在 rank-2)
        - precision=2/3
        - recall=2/2=1.0
        """
        inp = make_eval_input(["c", "a", "b"], ["a", "b"])
        out = self.ev.evaluate(inp)
        assert out.metrics["hit_rate"] == 1.0
        assert out.metrics["mrr"] == pytest.approx(0.5)
        assert out.metrics["precision"] == pytest.approx(2 / 3, abs=1e-5)
        assert out.metrics["recall"] == pytest.approx(1.0)

    def test_combined_all_empty(self):
        inp = make_eval_input([], [])
        out = self.ev.evaluate(inp)
        assert out.metrics == {"hit_rate": 0.0, "mrr": 0.0, "precision": 0.0, "recall": 0.0}

    def test_backend_name(self):
        assert self.ev.backend == "custom"

    def test_output_backend_matches(self):
        inp = make_eval_input(["a"], ["a"])
        out = self.ev.evaluate(inp)
        assert out.backend == "custom"

    def test_output_meta_contains_counts(self):
        inp = make_eval_input(["a", "b"], ["a"])
        out = self.ev.evaluate(inp)
        assert out.meta["retrieved_count"] == 2
        assert out.meta["golden_count"] == 1


# ---------------------------------------------------------------------------
# 4. CustomEvaluator — 错误处理与 TraceContext
# ---------------------------------------------------------------------------


class TestCustomEvaluatorErrors:
    def setup_method(self):
        self.ev = CustomEvaluator()

    def test_eval_input_none_raises_value_error(self):
        with pytest.raises(ValueError, match="eval_input"):
            self.ev.evaluate(None)  # type: ignore[arg-type]

    def test_trace_accepted_and_result_unchanged(self):
        """传入 trace 不影响计算结果。"""
        trace = TraceContext(trace_id="test-123", operation="evaluate")
        inp = make_eval_input(["a", "b"], ["a"])
        out_no_trace = self.ev.evaluate(inp)
        out_with_trace = self.ev.evaluate(inp, trace=trace)
        assert out_no_trace.metrics == out_with_trace.metrics

    def test_trace_none_is_ok(self):
        inp = make_eval_input(["a"], ["a"])
        out = self.ev.evaluate(inp, trace=None)
        assert out.metrics["hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# 5. 确定性验证
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self):
        ev = CustomEvaluator()
        inp = make_eval_input(["x", "a", "b"], ["a", "b"])
        results = [ev.evaluate(inp).metrics for _ in range(10)]
        for r in results[1:]:
            assert r == results[0], "相同输入应始终产生相同输出"

    def test_order_sensitive_mrr(self):
        """交换检索顺序应改变 MRR 结果。"""
        ev = CustomEvaluator()
        inp1 = make_eval_input(["a", "b", "x"], ["a"])
        inp2 = make_eval_input(["b", "a", "x"], ["a"])
        assert ev.evaluate(inp1).metrics["mrr"] == pytest.approx(1.0)
        assert ev.evaluate(inp2).metrics["mrr"] == pytest.approx(0.5)

    def test_duplicate_in_retrieved_handled_correctly(self):
        """retrieved_ids 中有重复 ID 时，命中计重复（precision 可超 recall）。"""
        ev = CustomEvaluator()
        inp = make_eval_input(["a", "a"], ["a"])
        out = ev.evaluate(inp)
        # 2 hits / 2 retrieved = precision=1.0; 即使有重复命中
        assert out.metrics["precision"] == pytest.approx(1.0)
        assert out.metrics["hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# 6. EvaluatorFactory 注册与路由测试
# ---------------------------------------------------------------------------


class FakeEvaluatorA(BaseEvaluator):
    """Fake evaluator A，始终返回 hit_rate=1.0。"""

    def __init__(self, settings=None):
        pass

    @property
    def backend(self) -> str:
        return "fake_a"

    def evaluate(self, eval_input, trace=None):
        return EvalOutput(backend=self.backend, metrics={"hit_rate": 1.0})


class FakeEvaluatorB(BaseEvaluator):
    """Fake evaluator B，始终返回 hit_rate=0.0。"""

    def __init__(self, settings=None):
        pass

    @property
    def backend(self) -> str:
        return "fake_b"

    def evaluate(self, eval_input, trace=None):
        return EvalOutput(backend=self.backend, metrics={"hit_rate": 0.0})


class TestEvaluatorFactory:
    def test_register_and_create(self):
        EvaluatorFactory.register("fake_a", FakeEvaluatorA)
        s = make_settings("fake_a")
        ev = EvaluatorFactory.create(s)
        assert isinstance(ev, FakeEvaluatorA)
        assert ev.backend == "fake_a"

    def test_registered_backends_lists_registered(self):
        EvaluatorFactory.register("fake_a", FakeEvaluatorA)
        EvaluatorFactory.register("fake_b", FakeEvaluatorB)
        backends = EvaluatorFactory.registered_backends()
        assert "fake_a" in backends
        assert "fake_b" in backends

    def test_unknown_backend_raises_evaluator_error(self):
        s = make_settings("not_exist")
        with pytest.raises(EvaluatorError) as exc_info:
            EvaluatorFactory.create(s)
        msg = str(exc_info.value)
        assert "not_exist" in msg

    def test_error_message_contains_registered_list(self):
        EvaluatorFactory.register("fake_a", FakeEvaluatorA)
        s = make_settings("unknown_backend")
        with pytest.raises(EvaluatorError) as exc_info:
            EvaluatorFactory.create(s)
        msg = str(exc_info.value)
        assert "fake_a" in msg

    def test_backend_name_is_case_insensitive(self):
        EvaluatorFactory.register("FAKE_A", FakeEvaluatorA)
        s = make_settings("fake_a")
        ev = EvaluatorFactory.create(s)
        assert isinstance(ev, FakeEvaluatorA)

    def test_create_calls_constructor_with_settings(self):
        """验证工厂以 evaluator_class(settings) 形式实例化。"""
        received_settings = {}

        class TrackingEvaluator(BaseEvaluator):
            def __init__(self, settings=None):
                received_settings["s"] = settings

            @property
            def backend(self):
                return "tracking"

            def evaluate(self, eval_input, trace=None):
                return EvalOutput(backend=self.backend, metrics={})

        EvaluatorFactory.register("tracking", TrackingEvaluator)
        s = make_settings("tracking")
        EvaluatorFactory.create(s)
        assert received_settings["s"] is s

    def test_multi_backend_routing(self):
        EvaluatorFactory.register("fake_a", FakeEvaluatorA)
        EvaluatorFactory.register("fake_b", FakeEvaluatorB)

        ev_a = EvaluatorFactory.create(make_settings("fake_a"))
        ev_b = EvaluatorFactory.create(make_settings("fake_b"))

        inp = make_eval_input(["x"], ["y"])
        out_a = ev_a.evaluate(inp)
        out_b = ev_b.evaluate(inp)

        assert out_a.metrics["hit_rate"] == 1.0
        assert out_b.metrics["hit_rate"] == 0.0

    def test_registry_isolation_between_tests(self):
        """注册表被 autouse fixture 清空，不污染其他测试。"""
        assert EvaluatorFactory.registered_backends() == []


# ---------------------------------------------------------------------------
# 7. CustomEvaluator 通过 Factory 注册后的集成路径
# ---------------------------------------------------------------------------


class TestCustomEvaluatorViaFactory:
    def test_register_custom_and_create(self):
        EvaluatorFactory.register("custom", CustomEvaluator)
        s = make_settings("custom")
        ev = EvaluatorFactory.create(s)
        assert isinstance(ev, CustomEvaluator)

    def test_end_to_end_via_factory(self):
        """完整路径：注册 → 创建 → 评估 → 验证 metrics。"""
        EvaluatorFactory.register("custom", CustomEvaluator)
        s = make_settings("custom")
        ev = EvaluatorFactory.create(s)

        inp = EvalInput(
            query="什么是RAG？",
            retrieved_ids=["chunk_1", "chunk_3", "chunk_5"],
            golden_ids=["chunk_1", "chunk_5"],
        )
        out = ev.evaluate(inp)

        assert out.backend == "custom"
        assert out.metrics["hit_rate"] == 1.0
        assert out.metrics["mrr"] == pytest.approx(1.0)  # chunk_1 在 rank-1
        assert out.metrics["precision"] == pytest.approx(2 / 3, abs=1e-5)
        assert out.metrics["recall"] == pytest.approx(1.0)

    def test_stable_metrics_repeated_calls(self):
        """多次调用同一 evaluator 实例，输出保持稳定。"""
        EvaluatorFactory.register("custom", CustomEvaluator)
        ev = EvaluatorFactory.create(make_settings("custom"))
        inp = make_eval_input(["a", "b", "c"], ["b", "c"])

        results = [ev.evaluate(inp).metrics for _ in range(5)]
        for r in results[1:]:
            assert r == results[0]


# ---------------------------------------------------------------------------
# 8. EvaluatorSettings 集成
# ---------------------------------------------------------------------------


class TestEvaluatorSettings:
    def test_default_settings_has_evaluator(self):
        s = Settings()
        assert hasattr(s, "evaluator")
        assert s.evaluator.backend == "custom"

    def test_evaluator_backend_changeable(self):
        s = Settings()
        s.evaluator.backend = "ragas"
        assert s.evaluator.backend == "ragas"

    def test_evaluator_backends_default_is_list(self):
        s = Settings()
        assert isinstance(s.evaluator.backends, list)
        assert "custom" in s.evaluator.backends
