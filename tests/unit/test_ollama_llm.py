"""Unit tests for B7.2: Ollama LLM 实现（mock HTTP，不走真实网络）

验收标准
--------
- provider=ollama 时 LLMFactory 路由到 OllamaLLM。
- chat() 正常调用返回模型回复内容。
- chat() 传入 None messages → LLMError，错误信息包含 provider 名称。
- chat() 传入空列表 → LLMError，错误信息包含 provider 名称。
- APIConnectionError（Ollama 未启动）→ LLMError，包含 base_url 与提示信息。
- APITimeoutError → LLMError，包含超时提示。
- NotFoundError（模型未下载）→ LLMError，包含模型名称与 ollama pull 提示。
- APIStatusError → LLMError，包含状态码。
- 默认 base_url 使用 ollama_base_url（http://localhost:11434/v1）。
- ollama_model 字段正确决定使用的模型名称。
- settings.llm.base_url 可覆盖 ollama_base_url。
- provider 属性返回 "ollama"。
- 全部测试不走真实网络（通过 unittest.mock.patch 注入假响应）。
"""

from __future__ import annotations

import pytest
from typing import List
from unittest.mock import MagicMock, patch

from src.libs.llm.base_llm import BaseLLM, LLMError, Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.ollama_llm import OllamaLLM
from src.core.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(
    provider: str = "ollama",
    model: str = "gpt-4o",          # 保持与其他 provider 一致的默认值
    api_key: str | None = None,
    base_url: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.1:8b",
) -> Settings:
    s = Settings()
    s.llm.provider = provider
    s.llm.model = model
    s.llm.api_key = api_key
    s.llm.base_url = base_url
    s.llm.ollama_base_url = ollama_base_url
    s.llm.ollama_model = ollama_model
    return s


def make_fake_completion(content: str = "hello from ollama") -> MagicMock:
    """构造一个符合 openai.ChatCompletion 结构的假响应。"""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def make_messages(text: str = "Hello") -> List[Message]:
    return [Message(role="user", content=text)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试前后清理 LLMFactory 注册表，防止测试间互相污染。"""
    LLMFactory._custom_registry.clear()
    yield
    LLMFactory._custom_registry.clear()


# ---------------------------------------------------------------------------
# OllamaLLM 基本属性测试
# ---------------------------------------------------------------------------


class TestOllamaLLMBasic:
    """OllamaLLM 基本属性与初始化测试。"""

    def test_provider_name(self):
        """provider 属性应返回 'ollama'。"""
        with patch("openai.OpenAI"):
            llm = OllamaLLM(make_settings())
        assert llm.provider == "ollama"

    def test_is_base_llm_subclass(self):
        """OllamaLLM 应是 BaseLLM 的子类。"""
        assert issubclass(OllamaLLM, BaseLLM)

    def test_default_base_url_uses_ollama_base_url(self):
        """默认情况下应使用 ollama_base_url + /v1。"""
        with patch("openai.OpenAI") as MockClient:
            OllamaLLM(make_settings(ollama_base_url="http://localhost:11434"))
        _, kwargs = MockClient.call_args
        assert "base_url" in kwargs
        assert kwargs["base_url"] == "http://localhost:11434/v1"

    def test_custom_base_url_overrides_ollama_base_url(self):
        """settings.llm.base_url 应覆盖 ollama_base_url。"""
        with patch("openai.OpenAI") as MockClient:
            OllamaLLM(make_settings(base_url="http://remote-ollama:11434"))
        _, kwargs = MockClient.call_args
        assert kwargs["base_url"] == "http://remote-ollama:11434/v1"

    def test_base_url_with_trailing_v1_not_duplicated(self):
        """若 base_url 已包含 /v1 后缀，不应重复追加。"""
        with patch("openai.OpenAI") as MockClient:
            OllamaLLM(make_settings(base_url="http://localhost:11434/v1"))
        _, kwargs = MockClient.call_args
        assert kwargs["base_url"] == "http://localhost:11434/v1"

    def test_base_url_with_trailing_slash(self):
        """base_url 末尾带斜杠时应正确拼接为 .../v1。"""
        with patch("openai.OpenAI") as MockClient:
            OllamaLLM(make_settings(ollama_base_url="http://localhost:11434/"))
        _, kwargs = MockClient.call_args
        assert kwargs["base_url"] == "http://localhost:11434/v1"

    def test_model_uses_ollama_model_field(self):
        """模型名称应使用 settings.llm.ollama_model 字段。"""
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("ok")
            MockClient.return_value.chat.completions.create.return_value = fake_resp
            llm = OllamaLLM(make_settings(ollama_model="mistral:7b"))
            llm.chat(make_messages())
        _, kwargs = MockClient.return_value.chat.completions.create.call_args
        assert kwargs.get("model") == "mistral:7b"

    def test_api_key_is_placeholder(self):
        """Ollama 不需要真实 API Key，客户端应使用 'ollama' 占位符。"""
        with patch("openai.OpenAI") as MockClient:
            OllamaLLM(make_settings())
        _, kwargs = MockClient.call_args
        assert kwargs["api_key"] == "ollama"


# ---------------------------------------------------------------------------
# chat() 正常调用测试
# ---------------------------------------------------------------------------


class TestOllamaLLMChat:
    """OllamaLLM.chat() 正常最终场景测试。"""

    def test_chat_returns_content(self):
        """正常调用应返回 choices[0].message.content。"""
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("你好，我是 Ollama!")
            MockClient.return_value.chat.completions.create.return_value = fake_resp
            llm = OllamaLLM(make_settings())
            result = llm.chat(make_messages("你好"))
        assert result == "你好，我是 Ollama!"

    def test_chat_passes_messages_correctly(self):
        """chat() 应将 Messages 转换为 dict 格式传给 API。"""
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("reply")
            MockClient.return_value.chat.completions.create.return_value = fake_resp
            llm = OllamaLLM(make_settings())
            msgs = [
                Message(role="system", content="你是一个助手"),
                Message(role="user", content="你好"),
            ]
            llm.chat(msgs)

        _, kwargs = MockClient.return_value.chat.completions.create.call_args
        assert kwargs["messages"] == [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"},
        ]

    def test_chat_passes_extra_kwargs(self):
        """chat() 应将 **kwargs 透传给底层 API。"""
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("ok")
            MockClient.return_value.chat.completions.create.return_value = fake_resp
            llm = OllamaLLM(make_settings())
            llm.chat(make_messages(), temperature=0.5, max_tokens=100)

        _, kwargs = MockClient.return_value.chat.completions.create.call_args
        assert kwargs.get("temperature") == 0.5
        assert kwargs.get("max_tokens") == 100

    def test_chat_none_content_raises_llm_error(self):
        """API 返回 None content 时应抛出 LLMError。"""
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion(None)
            MockClient.return_value.chat.completions.create.return_value = fake_resp
            llm = OllamaLLM(make_settings())
        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        assert "ollama" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# chat() 参数校验测试
# ---------------------------------------------------------------------------


class TestOllamaLLMValidation:
    """OllamaLLM.chat() 参数校验测试。"""

    def test_chat_none_messages_raises_llm_error(self):
        """messages=None 时应抛出 LLMError，错误信息包含 provider 名称。"""
        with patch("openai.OpenAI"):
            llm = OllamaLLM(make_settings())
        with pytest.raises(LLMError) as exc_info:
            llm.chat(None)  # type: ignore[arg-type]
        err = str(exc_info.value)
        assert "ollama" in err
        assert "None" in err

    def test_chat_empty_messages_raises_llm_error(self):
        """messages=[] 时应抛出 LLMError，错误信息包含 provider 名称。"""
        with patch("openai.OpenAI"):
            llm = OllamaLLM(make_settings())
        with pytest.raises(LLMError) as exc_info:
            llm.chat([])
        assert "ollama" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 错误处理测试
# ---------------------------------------------------------------------------


class TestOllamaLLMErrorHandling:
    """OllamaLLM 错误处理测试（连接失败、超时、模型未找到等）。"""

    def test_connection_error_raises_llm_error_with_url(self):
        """APIConnectionError 应被转换为 LLMError，包含 base_url 提示。"""
        import openai as _openai

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.side_effect = (
                _openai.APIConnectionError(request=MagicMock())
            )
            llm = OllamaLLM(make_settings())

        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        err = str(exc_info.value)
        assert "ollama" in err.lower()
        assert "localhost" in err or "11434" in err or "连接" in err

    def test_timeout_error_raises_llm_error(self):
        """APITimeoutError 应被转换为 LLMError，包含超时提示。"""
        import openai as _openai

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.side_effect = (
                _openai.APITimeoutError(request=MagicMock())
            )
            llm = OllamaLLM(make_settings())

        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        err = str(exc_info.value)
        assert "ollama" in err.lower()
        assert "超时" in err or "timeout" in err.lower()

    def test_not_found_error_raises_llm_error_with_model_hint(self):
        """模型不存在（NotFoundError）应抛出 LLMError，包含模型名称和 pull 提示。"""
        import openai as _openai

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.side_effect = (
                _openai.NotFoundError(
                    "model not found",
                    response=MagicMock(status_code=404),
                    body={"error": {"message": "model not found"}},
                )
            )
            llm = OllamaLLM(make_settings(ollama_model="nonexistent-model:7b"))

        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        err = str(exc_info.value)
        assert "ollama" in err.lower()
        assert "nonexistent-model:7b" in err
        assert "pull" in err.lower() or "下载" in err

    def test_api_status_error_raises_llm_error_with_status_code(self):
        """APIStatusError 应被转换为 LLMError，包含状态码。"""
        import openai as _openai

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.side_effect = (
                _openai.APIStatusError(
                    "Internal Server Error",
                    response=MagicMock(status_code=500),
                    body={"error": {"message": "Internal Server Error"}},
                )
            )
            llm = OllamaLLM(make_settings())

        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        err = str(exc_info.value)
        assert "ollama" in err.lower()
        assert "500" in err

    def test_generic_exception_raises_llm_error(self):
        """其他未知异常也应被转换为 LLMError。"""
        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.side_effect = (
                RuntimeError("unexpected error")
            )
            llm = OllamaLLM(make_settings())

        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        assert "ollama" in str(exc_info.value).lower()

    def test_missing_openai_sdk_raises_llm_error(self):
        """未安装 openai SDK 时 __init__ 应抛出 LLMError。"""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(LLMError) as exc_info:
                OllamaLLM(make_settings())
        assert "ollama" in str(exc_info.value).lower()
        assert "openai" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# LLMFactory 集成测试
# ---------------------------------------------------------------------------


class TestOllamaLLMFactory:
    """OllamaLLM 与 LLMFactory 集成测试。"""

    def test_factory_routes_to_ollama(self):
        """provider=ollama 时 LLMFactory 应路由到 OllamaLLM。"""
        with patch("openai.OpenAI"):
            LLMFactory.register("ollama", OllamaLLM)
            llm = LLMFactory.create(make_settings(provider="ollama"))
        assert isinstance(llm, OllamaLLM)
        assert llm.provider == "ollama"

    def test_factory_ollama_e2e(self):
        """端到端：通过工厂创建 OllamaLLM 后调用 chat()。"""
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("smoke ok ollama")
            MockClient.return_value.chat.completions.create.return_value = fake_resp

            LLMFactory.register("ollama", OllamaLLM)
            llm = LLMFactory.create(make_settings(provider="ollama"))
            result = llm.chat(make_messages("smoke test"))

        assert result == "smoke ok ollama"

    def test_factory_all_four_providers_simultaneously(self):
        """同时注册四个 provider，按 settings 路由互不干扰。"""
        from src.libs.llm.openai_llm import OpenAILLM
        from src.libs.llm.azure_llm import AzureLLM
        from src.libs.llm.deepseek_llm import DeepSeekLLM

        with patch("openai.OpenAI"), patch("openai.AzureOpenAI"):
            LLMFactory.register("openai", OpenAILLM)
            LLMFactory.register("azure", AzureLLM)
            LLMFactory.register("deepseek", DeepSeekLLM)
            LLMFactory.register("ollama", OllamaLLM)

            s_oa = Settings(); s_oa.llm.provider = "openai"; s_oa.llm.api_key = "k"
            s_az = Settings(); s_az.llm.provider = "azure"; s_az.llm.azure_api_key = "k"
            s_az.llm.azure_endpoint = "https://test.openai.azure.com/"
            s_ds = Settings(); s_ds.llm.provider = "deepseek"; s_ds.llm.api_key = "k"
            s_ol = Settings(); s_ol.llm.provider = "ollama"

            llm_oa = LLMFactory.create(s_oa)
            llm_az = LLMFactory.create(s_az)
            llm_ds = LLMFactory.create(s_ds)
            llm_ol = LLMFactory.create(s_ol)

        assert isinstance(llm_oa, OpenAILLM)
        assert isinstance(llm_az, AzureLLM)
        assert isinstance(llm_ds, DeepSeekLLM)
        assert isinstance(llm_ol, OllamaLLM)
