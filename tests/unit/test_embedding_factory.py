"""Unit tests for B2: Embedding 抽象接口与工厂

验收标准
--------
- Fake embedding 返回稳定（确定性）向量。
- EmbeddingFactory 按 provider 正确分流。
- 未知 provider 时 EmbeddingError 消息包含 provider 名称与已注册列表。
- embed() 空输入抛出 ValueError。
- embed_one() 便捷方法正常工作。
"""

import pytest

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.core.settings import Settings


# ---------------------------------------------------------------------------
# Fake Embedding（行内 Stub，不依赖任何外部服务）
# ---------------------------------------------------------------------------

_FAKE_DIM = 8  # Fake 向量维度（测试用，足够小且稳定）


class FakeEmbedding(BaseEmbedding):
    """确定性 Fake Embedding：向量值由文本内容哈希决定，保证稳定可重现。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.call_count = 0
        self.last_texts: list = []

    @property
    def provider(self) -> str:
        return "fake"

    def embed(self, texts, trace=None, **kwargs):
        if not texts:
            raise ValueError("texts 不能为空")
        self.call_count += 1
        self.last_texts = list(texts)
        # 确定性向量：每个文本生成固定的 _FAKE_DIM 维向量
        result = []
        for text in texts:
            seed = hash(text) % 1000
            vec = [float((seed + i) % 100) / 100.0 for i in range(_FAKE_DIM)]
            result.append(vec)
        return result


class AnotherFakeEmbedding(BaseEmbedding):
    """另一个 Fake，用于验证多 provider 路由。"""

    def __init__(self, settings: Settings) -> None:
        pass

    @property
    def provider(self) -> str:
        return "another_fake"

    def embed(self, texts, trace=None, **kwargs):
        return [[0.0] * _FAKE_DIM for _ in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试前后清理注册表，防止污染。"""
    EmbeddingFactory._custom_registry.clear()
    yield
    EmbeddingFactory._custom_registry.clear()


@pytest.fixture()
def fake_settings() -> Settings:
    s = Settings()
    s.embedding.provider = "fake"
    return s


@pytest.fixture()
def unknown_settings() -> Settings:
    s = Settings()
    s.embedding.provider = "unknown_provider_xyz"
    return s


# ---------------------------------------------------------------------------
# BaseEmbedding 接口约束
# ---------------------------------------------------------------------------


class TestBaseEmbeddingContract:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseEmbedding()  # type: ignore[abstract]

    def test_fake_embedding_satisfies_interface(self, fake_settings):
        emb = FakeEmbedding(fake_settings)
        assert isinstance(emb, BaseEmbedding)

    def test_provider_returns_correct_name(self, fake_settings):
        emb = FakeEmbedding(fake_settings)
        assert emb.provider == "fake"


# ---------------------------------------------------------------------------
# EmbeddingFactory 路由测试
# ---------------------------------------------------------------------------


class TestEmbeddingFactory:
    def test_register_and_create_fake_provider(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        assert isinstance(emb, FakeEmbedding)

    def test_create_returns_base_embedding(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        assert isinstance(emb, BaseEmbedding)

    def test_factory_routes_to_correct_class(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        EmbeddingFactory.register("another_fake", AnotherFakeEmbedding)

        fake_settings.embedding.provider = "another_fake"
        emb = EmbeddingFactory.create(fake_settings)
        assert isinstance(emb, AnotherFakeEmbedding)

    def test_unknown_provider_raises_embedding_error(self, unknown_settings):
        with pytest.raises(EmbeddingError) as exc_info:
            EmbeddingFactory.create(unknown_settings)
        assert "unknown_provider_xyz" in str(exc_info.value)

    def test_error_message_lists_registered_providers(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        fake_settings.embedding.provider = "not_exist"
        with pytest.raises(EmbeddingError) as exc_info:
            EmbeddingFactory.create(fake_settings)
        # 错误信息应列出 known providers
        assert "fake" in str(exc_info.value)

    def test_provider_name_case_insensitive(self, fake_settings):
        EmbeddingFactory.register("FAKE_UPPER", FakeEmbedding)
        fake_settings.embedding.provider = "fake_upper"
        emb = EmbeddingFactory.create(fake_settings)
        assert isinstance(emb, FakeEmbedding)
        EmbeddingFactory._custom_registry.pop("fake_upper", None)

    def test_registered_providers_returns_list(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        providers = EmbeddingFactory.registered_providers()
        assert isinstance(providers, list)
        assert "fake" in providers

    def test_register_overwrites_existing_provider(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        EmbeddingFactory.register("fake", AnotherFakeEmbedding)  # overwrite
        emb = EmbeddingFactory.create(fake_settings)
        assert isinstance(emb, AnotherFakeEmbedding)


# ---------------------------------------------------------------------------
# BaseEmbedding.embed() 行为测试
# ---------------------------------------------------------------------------


class TestFakeEmbeddingBehavior:
    def test_embed_returns_list_of_vectors(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        result = emb.embed(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_embed_vector_length_matches_dim(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        result = emb.embed(["test text"])
        assert len(result[0]) == _FAKE_DIM

    def test_embed_returns_floats(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        result = emb.embed(["check types"])
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_is_deterministic(self, fake_settings):
        """相同输入应产生相同向量（稳定性要求）。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        text = "deterministic test"
        v1 = emb.embed([text])[0]
        v2 = emb.embed([text])[0]
        assert v1 == v2

    def test_embed_different_texts_produce_different_vectors(self, fake_settings):
        """不同文本应产生不同向量。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        v1 = emb.embed(["text A"])[0]
        v2 = emb.embed(["text B"])[0]
        assert v1 != v2

    def test_embed_output_length_matches_input(self, fake_settings):
        """输出长度应与输入列表等长。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        texts = ["a", "b", "c", "d", "e"]
        result = emb.embed(texts)
        assert len(result) == len(texts)

    def test_embed_empty_texts_raises_value_error(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        with pytest.raises(ValueError):
            emb.embed([])

    def test_embed_increments_call_count(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb: FakeEmbedding = EmbeddingFactory.create(fake_settings)  # type: ignore[assignment]
        emb.embed(["x"])
        emb.embed(["y"])
        assert emb.call_count == 2

    def test_embed_records_last_texts(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb: FakeEmbedding = EmbeddingFactory.create(fake_settings)  # type: ignore[assignment]
        emb.embed(["track me"])
        assert emb.last_texts == ["track me"]

    def test_embed_with_trace_param_accepted(self, fake_settings):
        """trace 参数应被接受（即使 Fake 不使用它）。"""
        from src.core.types import TraceContext
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        trace = TraceContext(trace_id="t1", operation="embed_test")
        result = emb.embed(["trace test"], trace=trace)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# embed_one() 便捷方法测试
# ---------------------------------------------------------------------------


class TestEmbedOne:
    def test_embed_one_returns_single_vector(self, fake_settings):
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        vec = emb.embed_one("single text")
        assert isinstance(vec, list)
        assert len(vec) == _FAKE_DIM

    def test_embed_one_consistent_with_embed(self, fake_settings):
        """embed_one(text) 应与 embed([text])[0] 结果相同。"""
        EmbeddingFactory.register("fake", FakeEmbedding)
        emb = EmbeddingFactory.create(fake_settings)
        text = "consistency check"
        assert emb.embed_one(text) == emb.embed([text])[0]
