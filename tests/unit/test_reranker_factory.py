"""Unit tests for B5: Reranker 抽象接口与工厂（含 None 回退）

验收标准
--------
- NoneReranker 保持候选列表原顺序不变。
- backend=none 时 RerankerFactory.create() 返回 NoneReranker 实例（无需注册）。
- 自定义 backend 注册后可通过 Factory 路由创建。
- 未知 backend 时 RerankerError 消息包含 backend 名称与已注册列表。
- rerank() 空列表输入返回空列表（不报错）。
- rerank() candidates=None 时抛出 ValueError。
- rerank() 接受可选 trace 参数。
- 多 backend 注册后正确按 settings 路由。
- RerankerFactory 不硬编码 None backend：始终作为内置默认可用。
"""

import pytest
from typing import Any, List, Optional

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankerError
from src.libs.reranker.reranker_factory import RerankerFactory
from src.core.settings import Settings
from src.core.types import TraceContext


# ---------------------------------------------------------------------------
# Fake Rerankers（行内 Stub，不依赖任何外部依赖）
# ---------------------------------------------------------------------------


class ScoreBasedFakeReranker(BaseReranker):
    """按 score 属性降序排列的确定性 Fake Reranker。

    要求 candidates 元素具有 ``score`` 属性（dict 或对象）。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.call_count = 0
        self.last_query: str = ""

    @property
    def backend(self) -> str:
        return "score_fake"

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        trace: Optional[TraceContext] = None,
    ) -> List[Any]:
        if candidates is None:
            raise ValueError("candidates 不能为 None")
        self.call_count += 1
        self.last_query = query
        # 按 score 字段降序排列（确定性）
        return sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)


class ReverseOrderFakeReranker(BaseReranker):
    """倒序 Fake Reranker，用于验证多 backend 路由。"""

    def __init__(self, settings: Settings) -> None:
        pass

    @property
    def backend(self) -> str:
        return "reverse_fake"

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        trace: Optional[TraceContext] = None,
    ) -> List[Any]:
        if candidates is None:
            raise ValueError("candidates 不能为 None")
        return list(reversed(candidates))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试前后清理注册表，防止污染。"""
    RerankerFactory._custom_registry.clear()
    yield
    RerankerFactory._custom_registry.clear()


@pytest.fixture()
def none_settings() -> Settings:
    s = Settings()
    s.reranker.backend = "none"
    return s


@pytest.fixture()
def score_fake_settings() -> Settings:
    s = Settings()
    s.reranker.backend = "score_fake"
    return s


@pytest.fixture()
def unknown_settings() -> Settings:
    s = Settings()
    s.reranker.backend = "unknown_backend_xyz"
    return s


@pytest.fixture()
def sample_candidates():
    """确定性候选列表，包含 score 字段。"""
    return [
        {"id": "c1", "text": "chunk one", "score": 0.5},
        {"id": "c2", "text": "chunk two", "score": 0.9},
        {"id": "c3", "text": "chunk three", "score": 0.3},
    ]


@pytest.fixture()
def trace_ctx() -> TraceContext:
    return TraceContext(trace_id="b5-test", operation="rerank_test")


# ---------------------------------------------------------------------------
# BaseReranker 接口约束
# ---------------------------------------------------------------------------


class TestBaseRerankerContract:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]

    def test_fake_reranker_satisfies_interface(self, none_settings):
        reranker = ScoreBasedFakeReranker(none_settings)
        assert isinstance(reranker, BaseReranker)

    def test_backend_returns_correct_name(self, none_settings):
        reranker = ScoreBasedFakeReranker(none_settings)
        assert reranker.backend == "score_fake"

    def test_none_reranker_is_base_reranker(self, none_settings):
        reranker = NoneReranker(none_settings)
        assert isinstance(reranker, BaseReranker)

    def test_none_reranker_backend_name(self, none_settings):
        reranker = NoneReranker(none_settings)
        assert reranker.backend == "none"


# ---------------------------------------------------------------------------
# NoneReranker 行为测试
# ---------------------------------------------------------------------------


class TestNoneReranker:
    def test_preserves_original_order(self, sample_candidates, none_settings):
        reranker = NoneReranker(none_settings)
        result = reranker.rerank("query", sample_candidates)
        assert [c["id"] for c in result] == ["c1", "c2", "c3"]

    def test_returns_shallow_copy(self, sample_candidates, none_settings):
        reranker = NoneReranker(none_settings)
        result = reranker.rerank("query", sample_candidates)
        # 结构相同但不是同一个对象
        assert result == sample_candidates
        assert result is not sample_candidates

    def test_empty_candidates_returns_empty_list(self, none_settings):
        reranker = NoneReranker(none_settings)
        result = reranker.rerank("query", [])
        assert result == []

    def test_candidates_none_raises_value_error(self, none_settings):
        reranker = NoneReranker(none_settings)
        with pytest.raises(ValueError):
            reranker.rerank("query", None)  # type: ignore[arg-type]

    def test_accepts_trace_argument(self, sample_candidates, none_settings, trace_ctx):
        reranker = NoneReranker(none_settings)
        result = reranker.rerank("query", sample_candidates, trace=trace_ctx)
        assert len(result) == len(sample_candidates)

    def test_accepts_none_settings(self, sample_candidates):
        reranker = NoneReranker(settings=None)
        result = reranker.rerank("query", sample_candidates)
        assert result == sample_candidates

    def test_single_candidate_preserved(self, none_settings):
        reranker = NoneReranker(none_settings)
        single = [{"id": "only", "text": "only chunk", "score": 1.0}]
        result = reranker.rerank("test query", single)
        assert result == single


# ---------------------------------------------------------------------------
# RerankerFactory 路由测试
# ---------------------------------------------------------------------------


class TestRerankerFactory:
    def test_none_backend_no_registration_needed(self, none_settings):
        """backend=none 无需注册，始终可通过 Factory 创建。"""
        reranker = RerankerFactory.create(none_settings)
        assert isinstance(reranker, NoneReranker)

    def test_none_backend_returns_none_reranker(self, none_settings):
        reranker = RerankerFactory.create(none_settings)
        assert isinstance(reranker, NoneReranker)

    def test_register_and_create_custom_backend(self, score_fake_settings):
        RerankerFactory.register("score_fake", ScoreBasedFakeReranker)
        reranker = RerankerFactory.create(score_fake_settings)
        assert isinstance(reranker, ScoreBasedFakeReranker)

    def test_create_returns_base_reranker(self, score_fake_settings):
        RerankerFactory.register("score_fake", ScoreBasedFakeReranker)
        reranker = RerankerFactory.create(score_fake_settings)
        assert isinstance(reranker, BaseReranker)

    def test_factory_routes_to_correct_class(self, none_settings):
        RerankerFactory.register("score_fake", ScoreBasedFakeReranker)
        RerankerFactory.register("reverse_fake", ReverseOrderFakeReranker)

        none_settings.reranker.backend = "reverse_fake"
        reranker = RerankerFactory.create(none_settings)
        assert isinstance(reranker, ReverseOrderFakeReranker)

    def test_unknown_backend_raises_reranker_error(self, unknown_settings):
        with pytest.raises(RerankerError) as exc_info:
            RerankerFactory.create(unknown_settings)
        assert "unknown_backend_xyz" in str(exc_info.value)

    def test_error_message_contains_backend_name(self, unknown_settings):
        with pytest.raises(RerankerError) as exc_info:
            RerankerFactory.create(unknown_settings)
        error_msg = str(exc_info.value)
        assert "unknown_backend_xyz" in error_msg

    def test_error_message_lists_none_as_available(self, unknown_settings):
        """错误信息中应包含内置的 none backend。"""
        with pytest.raises(RerankerError) as exc_info:
            RerankerFactory.create(unknown_settings)
        error_msg = str(exc_info.value)
        assert "none" in error_msg

    def test_error_message_lists_registered_backends(self, none_settings):
        RerankerFactory.register("score_fake", ScoreBasedFakeReranker)
        none_settings.reranker.backend = "no_exist"
        with pytest.raises(RerankerError) as exc_info:
            RerankerFactory.create(none_settings)
        assert "score_fake" in str(exc_info.value)

    def test_registered_backends_returns_custom_only(self):
        """registered_backends() 仅返回注册的自定义 backends，不含内置 none。"""
        RerankerFactory.register("score_fake", ScoreBasedFakeReranker)
        backends = RerankerFactory.registered_backends()
        assert "score_fake" in backends
        assert "none" not in backends

    def test_registry_is_isolated_between_tests(self):
        """autouse fixture 保证注册表在测试间隔离清理。"""
        assert RerankerFactory.registered_backends() == []

    def test_backend_name_case_insensitive(self, none_settings):
        """backend 名称应大小写不敏感。"""
        RerankerFactory.register("Score_Fake", ScoreBasedFakeReranker)
        none_settings.reranker.backend = "SCORE_FAKE"
        reranker = RerankerFactory.create(none_settings)
        assert isinstance(reranker, ScoreBasedFakeReranker)


# ---------------------------------------------------------------------------
# 端到端：NoneReranker 不影响已有排序
# ---------------------------------------------------------------------------


class TestNoneRerankerNoSideEffect:
    def test_none_reranker_does_not_change_order(self, none_settings):
        """backend=none 时候选排序与输入完全一致。"""
        candidates = [
            {"id": "high", "score": 0.9},
            {"id": "mid", "score": 0.5},
            {"id": "low", "score": 0.1},
        ]
        reranker = RerankerFactory.create(none_settings)
        result = reranker.rerank("test query", candidates)
        assert [c["id"] for c in result] == ["high", "mid", "low"]

    def test_none_reranker_vs_score_based_differ(self, none_settings):
        """NoneReranker 和 ScoreBasedFakeReranker 对同一输入产生不同输出。"""
        candidates = [
            {"id": "low", "score": 0.1},
            {"id": "high", "score": 0.9},
        ]
        none_reranker = NoneReranker(none_settings)
        none_result = none_reranker.rerank("query", candidates)

        fake_reranker = ScoreBasedFakeReranker(none_settings)
        fake_result = fake_reranker.rerank("query", candidates)

        # NoneReranker 保持原顺序（low 在前）
        assert none_result[0]["id"] == "low"
        # ScoreFake 按分数降序（high 在前）
        assert fake_result[0]["id"] == "high"

    def test_trace_context_passed_through(self, none_settings, trace_ctx, sample_candidates):
        """trace 参数可以传入，不影响功能（NoneReranker 不使用）。"""
        reranker = RerankerFactory.create(none_settings)
        result = reranker.rerank("query", sample_candidates, trace=trace_ctx)
        assert len(result) == len(sample_candidates)
