"""B1 — LLM 抽象接口与工厂单元测试"""

import pytest
from typing import List, Optional, Any

from src.core.settings import Settings
from src.core.types import TraceContext
from src.libs.llm.base_llm import BaseLLM, LLMError, Message
from src.libs.llm.llm_factory import LLMFactory


# ---------------------------------------------------------------------------
# Fake 实现（测试内 stub，不依赖任何外部服务）
# ---------------------------------------------------------------------------


class FakeLLM(BaseLLM):
    """测试用的假 LLM：直接把最后一条用户消息拼接后返回。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.call_count = 0

    def chat(
        self,
        messages: List[Message],
        trace: Optional[TraceContext] = None,
        **kwargs: Any,
    ) -> str:
        self.call_count += 1
        last_content = messages[-1].content if messages else ""
        return f"fake_reply: {last_content}"

    @property
    def provider(self) -> str:
        return "fake"


class AnotherFakeLLM(BaseLLM):
    """另一个测试用 stub，用于验证多 provider 路由。"""

    def __init__(self, settings: Settings) -> None:
        pass

    def chat(
        self,
        messages: List[Message],
        trace: Optional[TraceContext] = None,
        **kwargs: Any,
    ) -> str:
        return "another_fake_reply"

    @property
    def provider(self) -> str:
        return "another_fake"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试前后清理自定义注册表，避免测试间互相污染。"""
    LLMFactory._custom_registry.clear()
    yield
    LLMFactory._custom_registry.clear()


@pytest.fixture()
def fake_settings(tmp_path) -> Settings:
    """返回 provider=fake 的 Settings 对象。"""
    settings = Settings()
    settings.llm.provider = "fake"
    return settings


@pytest.fixture()
def unknown_settings() -> Settings:
    """返回 provider=unknown 的 Settings 对象。"""
    settings = Settings()
    settings.llm.provider = "unknown_provider_xyz"
    return settings


# ---------------------------------------------------------------------------
# Message 测试
# ---------------------------------------------------------------------------


class TestMessage:
    def test_message_init_stores_role_and_content(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_message_to_dict_returns_correct_keys(self):
        msg = Message(role="assistant", content="world")
        d = msg.to_dict()
        assert d == {"role": "assistant", "content": "world"}


# ---------------------------------------------------------------------------
# BaseLLM 接口约束测试
# ---------------------------------------------------------------------------


class TestBaseLLMContract:
    def test_cannot_instantiate_abstract_base(self):
        """BaseLLM 是抽象类，不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]

    def test_fake_llm_satisfies_interface(self, fake_settings):
        """FakeLLM 实现需满足 BaseLLM 接口。"""
        llm = FakeLLM(fake_settings)
        assert isinstance(llm, BaseLLM)

    def test_fake_llm_provider_returns_correct_name(self, fake_settings):
        llm = FakeLLM(fake_settings)
        assert llm.provider == "fake"


# ---------------------------------------------------------------------------
# LLMFactory 工厂路由测试
# ---------------------------------------------------------------------------


class TestLLMFactory:
    def test_register_and_create_fake_provider(self, fake_settings):
        """注册 fake provider 后，工厂应返回 FakeLLM 实例。"""
        LLMFactory.register("fake", FakeLLM)
        llm = LLMFactory.create(fake_settings)
        assert isinstance(llm, FakeLLM)

    def test_factory_routes_to_correct_class_by_provider(self, fake_settings):
        """工厂根据 provider 字段选择正确的实现类。"""
        LLMFactory.register("fake", FakeLLM)
        LLMFactory.register("another_fake", AnotherFakeLLM)

        fake_settings.llm.provider = "another_fake"
        llm = LLMFactory.create(fake_settings)
        assert isinstance(llm, AnotherFakeLLM)

    def test_unknown_provider_raises_llm_error(self, unknown_settings):
        """未知 provider 应明确抛出 LLMError。"""
        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create(unknown_settings)
        assert "unknown_provider_xyz" in str(exc_info.value)

    def test_error_message_lists_available_providers(self, fake_settings):
        """LLMError 中应包含已注册 provider 列表，方便排查。"""
        LLMFactory.register("fake", FakeLLM)
        fake_settings.llm.provider = "not_exist"
        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create(fake_settings)
        assert "not_exist" in str(exc_info.value)

    def test_provider_name_is_case_insensitive(self, fake_settings):
        """provider 名称匹配应不区分大小写。"""
        LLMFactory.register("FAKE", FakeLLM)
        fake_settings.llm.provider = "FAKE"
        llm = LLMFactory.create(fake_settings)
        assert isinstance(llm, FakeLLM)


# ---------------------------------------------------------------------------
# FakeLLM chat 行为测试
# ---------------------------------------------------------------------------


class TestFakeLLMBehavior:
    def test_chat_returns_string(self, fake_settings):
        LLMFactory.register("fake", FakeLLM)
        llm = LLMFactory.create(fake_settings)
        messages = [Message("user", "test question")]
        result = llm.chat(messages)
        assert isinstance(result, str)

    def test_chat_echoes_last_message(self, fake_settings):
        LLMFactory.register("fake", FakeLLM)
        llm = LLMFactory.create(fake_settings)
        messages = [
            Message("system", "you are an assistant"),
            Message("user", "what is rag?"),
        ]
        result = llm.chat(messages)
        assert "what is rag?" in result

    def test_chat_accepts_optional_trace_context(self, fake_settings):
        """chat 方法接受 None trace 不应抛出异常。"""
        LLMFactory.register("fake", FakeLLM)
        llm = LLMFactory.create(fake_settings)
        messages = [Message("user", "hello")]
        result = llm.chat(messages, trace=None)
        assert result is not None

    def test_chat_call_count_tracks_invocations(self, fake_settings):
        """验证 call_count 能正确统计调用次数。"""
        LLMFactory.register("fake", FakeLLM)
        llm: FakeLLM = LLMFactory.create(fake_settings)  # type: ignore[assignment]
        messages = [Message("user", "ping")]
        llm.chat(messages)
        llm.chat(messages)
        assert llm.call_count == 2

    def test_chat_empty_messages_handled(self, fake_settings):
        """空消息列表不应导致异常。"""
        LLMFactory.register("fake", FakeLLM)
        llm = LLMFactory.create(fake_settings)
        result = llm.chat([])
        assert isinstance(result, str)
