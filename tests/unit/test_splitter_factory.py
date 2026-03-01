"""Unit tests for B3: Splitter 抽象接口与工厂

验收标准
--------
- Factory 能根据配置返回不同类型的 Splitter 实例（使用 Fake 实现）。
- FakeSplitter 返回稳定（确定性）切分结果。
- SplitterFactory 按 provider 正确分流。
- 未知 provider 时 SplitterError 消息包含 provider 名称与已注册列表。
- split_text() 空输入抛出 ValueError。
- split_text() 接受可选 trace 参数。
"""

import pytest

from src.libs.splitter.base_splitter import BaseSplitter, SplitterError
from src.libs.splitter.splitter_factory import SplitterFactory
from src.core.settings import Settings


# ---------------------------------------------------------------------------
# Fake Splitter（行内 Stub，不依赖任何外部依赖）
# ---------------------------------------------------------------------------


class FakeSplitter(BaseSplitter):
    """确定性 Fake Splitter：按固定大小切分，结果稳定可重现。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.call_count = 0
        self.last_text: str = ""

    @property
    def provider(self) -> str:
        return "fake"

    def split_text(self, text: str, trace=None, **kwargs):
        if not text:
            raise ValueError("text 不能为空")
        self.call_count += 1
        self.last_text = text
        # 简单按 chunk_size 切割（确定性行为）
        chunk_size = self._settings.splitter.chunk_size
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]


class AnotherFakeSplitter(BaseSplitter):
    """另一个 Fake，用于验证多 provider 路由。"""

    def __init__(self, settings: Settings) -> None:
        pass

    @property
    def provider(self) -> str:
        return "another_fake"

    def split_text(self, text: str, trace=None, **kwargs):
        if not text:
            raise ValueError("text 不能为空")
        # 按单词切分
        return text.split()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试前后清理注册表，防止污染。"""
    SplitterFactory._custom_registry.clear()
    yield
    SplitterFactory._custom_registry.clear()


@pytest.fixture()
def fake_settings() -> Settings:
    s = Settings()
    s.splitter.provider = "fake"
    s.splitter.chunk_size = 100
    s.splitter.chunk_overlap = 20
    return s


@pytest.fixture()
def unknown_settings() -> Settings:
    s = Settings()
    s.splitter.provider = "unknown_provider_xyz"
    return s


# ---------------------------------------------------------------------------
# BaseSplitter 接口约束
# ---------------------------------------------------------------------------


class TestBaseSplitterContract:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseSplitter()  # type: ignore[abstract]

    def test_fake_splitter_satisfies_interface(self, fake_settings):
        splitter = FakeSplitter(fake_settings)
        assert isinstance(splitter, BaseSplitter)

    def test_provider_returns_correct_name(self, fake_settings):
        splitter = FakeSplitter(fake_settings)
        assert splitter.provider == "fake"


# ---------------------------------------------------------------------------
# SplitterFactory 路由测试
# ---------------------------------------------------------------------------


class TestSplitterFactory:
    def test_register_and_create_fake_provider(self, fake_settings):
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(fake_settings)
        assert isinstance(splitter, FakeSplitter)

    def test_create_returns_base_splitter(self, fake_settings):
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(fake_settings)
        assert isinstance(splitter, BaseSplitter)

    def test_factory_routes_to_correct_class(self, fake_settings):
        SplitterFactory.register("fake", FakeSplitter)
        SplitterFactory.register("another_fake", AnotherFakeSplitter)

        fake_settings.splitter.provider = "another_fake"
        splitter = SplitterFactory.create(fake_settings)
        assert isinstance(splitter, AnotherFakeSplitter)

    def test_unknown_provider_raises_splitter_error(self, unknown_settings):
        with pytest.raises(SplitterError) as exc_info:
            SplitterFactory.create(unknown_settings)
        assert "unknown_provider_xyz" in str(exc_info.value)

    def test_error_message_lists_registered_providers(self, fake_settings):
        SplitterFactory.register("fake", FakeSplitter)
        fake_settings.splitter.provider = "not_exist"
        with pytest.raises(SplitterError) as exc_info:
            SplitterFactory.create(fake_settings)
        # 错误信息应列出 known providers
        assert "fake" in str(exc_info.value)

    def test_provider_name_case_insensitive(self, fake_settings):
        SplitterFactory.register("FAKE_UPPER", FakeSplitter)
        fake_settings.splitter.provider = "fake_upper"
        splitter = SplitterFactory.create(fake_settings)
        assert isinstance(splitter, FakeSplitter)
        SplitterFactory._custom_registry.pop("fake_upper", None)

    def test_registered_providers_returns_list(self, fake_settings):
        SplitterFactory.register("fake", FakeSplitter)
        providers = SplitterFactory.registered_providers()
        assert isinstance(providers, list)
        assert "fake" in providers

    def test_register_overwrites_existing_provider(self, fake_settings):
        SplitterFactory.register("fake", FakeSplitter)
        SplitterFactory.register("fake", AnotherFakeSplitter)  # overwrite
        splitter = SplitterFactory.create(fake_settings)
        assert isinstance(splitter, AnotherFakeSplitter)

    def test_empty_registry_raises_splitter_error(self, fake_settings):
        # 不注册任何 provider 直接创建应报错
        with pytest.raises(SplitterError):
            SplitterFactory.create(fake_settings)


# ---------------------------------------------------------------------------
# BaseSplitter.split_text() 行为测试
# ---------------------------------------------------------------------------


class TestFakeSplitterBehavior:
    def test_split_text_returns_list_of_strings(self, fake_settings):
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(fake_settings)
        result = splitter.split_text("Hello World")
        assert isinstance(result, list)
        assert all(isinstance(chunk, str) for chunk in result)

    def test_split_short_text_returns_single_chunk(self, fake_settings):
        """短文本（小于 chunk_size）应作为整体返回。"""
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(fake_settings)
        short_text = "Short."
        result = splitter.split_text(short_text)
        assert len(result) == 1
        assert result[0] == short_text

    def test_split_long_text_produces_multiple_chunks(self, fake_settings):
        """超过 chunk_size 的文本应被切分为多个 chunk。"""
        fake_settings.splitter.chunk_size = 10
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(fake_settings)
        long_text = "A" * 35
        result = splitter.split_text(long_text)
        assert len(result) > 1

    def test_split_is_deterministic(self, fake_settings):
        """相同输入应产生相同切分结果（稳定性要求）。"""
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(fake_settings)
        text = "Deterministic split test text"
        r1 = splitter.split_text(text)
        r2 = splitter.split_text(text)
        assert r1 == r2

    def test_split_empty_text_raises_value_error(self, fake_settings):
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(fake_settings)
        with pytest.raises(ValueError):
            splitter.split_text("")

    def test_split_increments_call_count(self, fake_settings):
        SplitterFactory.register("fake", FakeSplitter)
        splitter: FakeSplitter = SplitterFactory.create(fake_settings)  # type: ignore[assignment]
        splitter.split_text("first call")
        splitter.split_text("second call")
        assert splitter.call_count == 2

    def test_split_records_last_text(self, fake_settings):
        SplitterFactory.register("fake", FakeSplitter)
        splitter: FakeSplitter = SplitterFactory.create(fake_settings)  # type: ignore[assignment]
        splitter.split_text("track me")
        assert splitter.last_text == "track me"

    def test_split_chunks_cover_full_text(self, fake_settings):
        """所有 chunk 拼接后应还原原始文本（无重叠的 FakeSplitter）。"""
        fake_settings.splitter.chunk_size = 10
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(fake_settings)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = splitter.split_text(text)
        assert "".join(result) == text

    def test_split_with_trace_param_accepted(self, fake_settings):
        """trace 参数应被接受（即使 Fake 不使用它）。"""
        from src.core.types import TraceContext
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(fake_settings)
        trace = TraceContext(trace_id="t1", operation="split_test")
        result = splitter.split_text("trace test text", trace=trace)
        assert len(result) >= 1

    def test_another_fake_splits_by_word(self, fake_settings):
        """AnotherFakeSplitter 按单词切分。"""
        SplitterFactory.register("another_fake", AnotherFakeSplitter)
        fake_settings.splitter.provider = "another_fake"
        splitter = SplitterFactory.create(fake_settings)
        result = splitter.split_text("hello world foo")
        assert result == ["hello", "world", "foo"]


# ---------------------------------------------------------------------------
# SplitterSettings 默认值测试
# ---------------------------------------------------------------------------


class TestSplitterSettings:
    def test_default_provider_is_recursive(self):
        s = Settings()
        assert s.splitter.provider == "recursive"

    def test_default_chunk_size(self):
        s = Settings()
        assert s.splitter.chunk_size == 1000

    def test_default_chunk_overlap(self):
        s = Settings()
        assert s.splitter.chunk_overlap == 200

    def test_splitter_settings_are_mutable(self):
        s = Settings()
        s.splitter.provider = "fake"
        assert s.splitter.provider == "fake"
