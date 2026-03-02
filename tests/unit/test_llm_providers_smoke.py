"""Unit tests for B7.1: OpenAI-Compatible LLM Providers（Smoke Tests）

验收标准
--------
- provider=openai  时 LLMFactory 路由到 OpenAILLM，chat() 返回正确内容。
- provider=azure   时 LLMFactory 路由到 AzureLLM，chat() 返回正确内容。
- provider=deepseek 时 LLMFactory 路由到 DeepSeekLLM，chat() 返回正确内容。
- chat() 传入 None messages → LLMError，错误信息包含 provider 名称。
- chat() 传入空列表   → LLMError，错误信息包含 provider 名称。
- API 调用失败（AuthenticationError / RateLimitError / Timeout / …）→ LLMError，信息可读。
- DeepSeek 默认使用 deepseek-chat 模型与 DeepSeek base_url。
- Azure 使用 AzureOpenAI 客户端（不同于 openai.OpenAI）。
- 全部测试不走真实网络（通过 unittest.mock.patch 注入假响应）。
"""

from __future__ import annotations

import pytest
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

from src.libs.llm.base_llm import BaseLLM, LLMError, Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.deepseek_llm import DeepSeekLLM
from src.core.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: str = "sk-test",
    base_url: str | None = None,
    azure_api_key: str | None = "az-test",
    azure_endpoint: str | None = "https://test.openai.azure.com/",
    azure_api_version: str = "2024-02-15-preview",
) -> Settings:
    s = Settings()
    s.llm.provider = provider
    s.llm.model = model
    s.llm.api_key = api_key
    s.llm.base_url = base_url
    s.llm.azure_api_key = azure_api_key
    s.llm.azure_endpoint = azure_endpoint
    s.llm.azure_api_version = azure_api_version
    return s


def make_fake_completion(content: str = "hello from mock") -> MagicMock:
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
# OpenAILLM Tests
# ---------------------------------------------------------------------------


class TestOpenAILLM:
    """OpenAI 提供商单元测试（全 mock，不走真实网络）。"""

    def test_provider_name(self):
        with patch("openai.OpenAI"):
            llm = OpenAILLM(make_settings(provider="openai"))
        assert llm.provider == "openai"

    def test_chat_returns_content(self):
        """正常调用返回 choices[0].message.content。"""
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("Hi there!")
            instance = MockClient.return_value
            instance.chat.completions.create.return_value = fake_resp
            llm = OpenAILLM(make_settings())
            result = llm.chat(make_messages("Hello"))
        assert result == "Hi there!"

    def test_chat_none_messages_raises_llm_error(self):
        with patch("openai.OpenAI"):
            llm = OpenAILLM(make_settings())
        with pytest.raises(LLMError) as exc_info:
            llm.chat(None)  # type: ignore[arg-type]
        assert "openai" in str(exc_info.value)
        assert "None" in str(exc_info.value)

    def test_chat_empty_messages_raises_llm_error(self):
        with patch("openai.OpenAI"):
            llm = OpenAILLM(make_settings())
        with pytest.raises(LLMError) as exc_info:
            llm.chat([])
        assert "openai" in str(exc_info.value)

    def test_chat_api_auth_error(self):
        """AuthenticationError 应被转换为 LLMError。"""
        import openai as _openai

        with patch("openai.OpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create.side_effect = _openai.AuthenticationError(
                "Incorrect API key",
                response=MagicMock(status_code=401),
                body={"error": {"message": "Incorrect API key"}},
            )
            llm = OpenAILLM(make_settings())
        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        assert "openai" in str(exc_info.value).lower()
        assert "api key" in str(exc_info.value).lower() or "key" in str(exc_info.value).lower()

    def test_chat_rate_limit_error(self):
        """RateLimitError 应被转换为 LLMError。"""
        import openai as _openai

        with patch("openai.OpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create.side_effect = _openai.RateLimitError(
                "Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={"error": {"message": "Rate limit exceeded"}},
            )
            llm = OpenAILLM(make_settings())
        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        assert "openai" in str(exc_info.value).lower()

    def test_chat_timeout_error(self):
        """APITimeoutError 应被转换为 LLMError。"""
        import openai as _openai
        import httpx

        with patch("openai.OpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create.side_effect = _openai.APITimeoutError(
                request=MagicMock()
            )
            llm = OpenAILLM(make_settings())
        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        assert "openai" in str(exc_info.value).lower() or "timeout" in str(exc_info.value).lower()

    def test_chat_with_trace_context_ok(self):
        """传入 trace 参数不影响返回结果。"""
        from src.core.types import TraceContext
        trace = TraceContext(trace_id="t1", operation="test_chat")
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("traced response")
            instance = MockClient.return_value
            instance.chat.completions.create.return_value = fake_resp
            llm = OpenAILLM(make_settings())
            result = llm.chat(make_messages(), trace=trace)
        assert result == "traced response"

    def test_chat_passes_kwargs_to_api(self):
        """kwargs 应透传给 API（如 temperature、max_tokens）。"""
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("ok")
            instance = MockClient.return_value
            instance.chat.completions.create.return_value = fake_resp
            llm = OpenAILLM(make_settings())
            llm.chat(make_messages(), temperature=0.5, max_tokens=100)
            call_kwargs = instance.chat.completions.create.call_args[1]
        assert call_kwargs.get("temperature") == 0.5
        assert call_kwargs.get("max_tokens") == 100

    def test_uses_custom_base_url(self):
        """base_url 应传递给 OpenAI 客户端。"""
        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value = MagicMock()
            llm = OpenAILLM(make_settings(base_url="https://my-proxy.example.com/v1"))
        init_kwargs = MockClient.call_args[1]
        assert init_kwargs.get("base_url") == "https://my-proxy.example.com/v1"

    def test_chat_content_none_raises_llm_error(self):
        """当 choices[0].message.content 为 None 时应抛出 LLMError。"""
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion(None)  # type: ignore[arg-type]
            instance = MockClient.return_value
            instance.chat.completions.create.return_value = fake_resp
            llm = OpenAILLM(make_settings())
        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        assert "None" in str(exc_info.value) or "空内容" in str(exc_info.value)

    def test_is_base_llm_subclass(self):
        with patch("openai.OpenAI"):
            llm = OpenAILLM(make_settings())
        assert isinstance(llm, BaseLLM)


# ---------------------------------------------------------------------------
# AzureLLM Tests
# ---------------------------------------------------------------------------


class TestAzureLLM:
    """Azure OpenAI 提供商单元测试（全 mock）。"""

    def test_provider_name(self):
        with patch("openai.AzureOpenAI"):
            llm = AzureLLM(make_settings(provider="azure"))
        assert llm.provider == "azure"

    def test_chat_returns_content(self):
        with patch("openai.AzureOpenAI") as MockClient:
            fake_resp = make_fake_completion("azure response")
            instance = MockClient.return_value
            instance.chat.completions.create.return_value = fake_resp
            llm = AzureLLM(make_settings(provider="azure"))
            result = llm.chat(make_messages())
        assert result == "azure response"

    def test_uses_azure_openai_client(self):
        """Azure 实现必须使用 AzureOpenAI 客户端，不是 openai.OpenAI。"""
        with patch("openai.AzureOpenAI") as MockAzure, patch("openai.OpenAI") as MockOpenAI:
            llm = AzureLLM(make_settings(provider="azure"))
        MockAzure.assert_called_once()
        MockOpenAI.assert_not_called()

    def test_azure_client_receives_endpoint(self):
        """azure_endpoint 应正确传递给 AzureOpenAI 客户端。"""
        with patch("openai.AzureOpenAI") as MockClient:
            MockClient.return_value = MagicMock()
            llm = AzureLLM(make_settings(
                provider="azure",
                azure_endpoint="https://my-resource.openai.azure.com/",
            ))
        init_kwargs = MockClient.call_args[1]
        assert init_kwargs.get("azure_endpoint") == "https://my-resource.openai.azure.com/"

    def test_azure_client_receives_api_version(self):
        """azure_api_version 应正确传递给 AzureOpenAI 客户端。"""
        with patch("openai.AzureOpenAI") as MockClient:
            MockClient.return_value = MagicMock()
            llm = AzureLLM(make_settings(
                provider="azure",
                azure_api_version="2024-05-01-preview",
            ))
        init_kwargs = MockClient.call_args[1]
        assert init_kwargs.get("api_version") == "2024-05-01-preview"

    def test_chat_none_messages_raises_llm_error(self):
        with patch("openai.AzureOpenAI"):
            llm = AzureLLM(make_settings(provider="azure"))
        with pytest.raises(LLMError) as exc_info:
            llm.chat(None)  # type: ignore[arg-type]
        assert "azure" in str(exc_info.value)

    def test_chat_empty_messages_raises_llm_error(self):
        with patch("openai.AzureOpenAI"):
            llm = AzureLLM(make_settings(provider="azure"))
        with pytest.raises(LLMError) as exc_info:
            llm.chat([])
        assert "azure" in str(exc_info.value)

    def test_chat_auth_error_wraps_to_llm_error(self):
        import openai as _openai

        with patch("openai.AzureOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create.side_effect = _openai.AuthenticationError(
                "Azure key invalid",
                response=MagicMock(status_code=401),
                body={"error": {"message": "Azure key invalid"}},
            )
            llm = AzureLLM(make_settings(provider="azure"))
        with pytest.raises(LLMError) as exc_info:
            llm.chat(make_messages())
        assert "azure" in str(exc_info.value).lower()

    def test_is_base_llm_subclass(self):
        with patch("openai.AzureOpenAI"):
            llm = AzureLLM(make_settings(provider="azure"))
        assert isinstance(llm, BaseLLM)


# ---------------------------------------------------------------------------
# DeepSeekLLM Tests
# ---------------------------------------------------------------------------


class TestDeepSeekLLM:
    """DeepSeek 提供商单元测试（全 mock）。"""

    def test_provider_name(self):
        with patch("openai.OpenAI"):
            llm = DeepSeekLLM(make_settings(provider="deepseek"))
        assert llm.provider == "deepseek"

    def test_chat_returns_content(self):
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("deepseek response")
            instance = MockClient.return_value
            instance.chat.completions.create.return_value = fake_resp
            llm = DeepSeekLLM(make_settings(provider="deepseek"))
            result = llm.chat(make_messages())
        assert result == "deepseek response"

    def test_default_model_is_deepseek_chat(self):
        """未配置 model 时，默认使用 deepseek-chat。"""
        s = make_settings(provider="deepseek", model="")  # 空 model → fallback 到 deepseek-chat
        s.llm.model = ""  # 故意留空
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("ok")
            instance = MockClient.return_value
            instance.chat.completions.create.return_value = fake_resp
            llm = DeepSeekLLM(s)
            assert llm._model == "deepseek-chat"

    def test_default_base_url_is_deepseek(self):
        """未配置 base_url 时，应使用 DeepSeek 默认端点。"""
        from src.libs.llm.deepseek_llm import _DEEPSEEK_BASE_URL
        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value = MagicMock()
            llm = DeepSeekLLM(make_settings(provider="deepseek", base_url=None))
        init_kwargs = MockClient.call_args[1]
        assert init_kwargs.get("base_url") == _DEEPSEEK_BASE_URL

    def test_custom_base_url_overrides_default(self):
        """配置了 base_url 时，应覆盖默认 DeepSeek 端点。"""
        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value = MagicMock()
            llm = DeepSeekLLM(make_settings(provider="deepseek", base_url="https://custom.api.com/v1"))
        init_kwargs = MockClient.call_args[1]
        assert init_kwargs.get("base_url") == "https://custom.api.com/v1"

    def test_chat_none_messages_raises_llm_error(self):
        with patch("openai.OpenAI"):
            llm = DeepSeekLLM(make_settings(provider="deepseek"))
        with pytest.raises(LLMError) as exc_info:
            llm.chat(None)  # type: ignore[arg-type]
        assert "deepseek" in str(exc_info.value)

    def test_chat_empty_messages_raises_llm_error(self):
        with patch("openai.OpenAI"):
            llm = DeepSeekLLM(make_settings(provider="deepseek"))
        with pytest.raises(LLMError) as exc_info:
            llm.chat([])
        assert "deepseek" in str(exc_info.value)

    def test_is_base_llm_subclass(self):
        with patch("openai.OpenAI"):
            llm = DeepSeekLLM(make_settings(provider="deepseek"))
        assert isinstance(llm, BaseLLM)


# ---------------------------------------------------------------------------
# LLMFactory Routing Tests
# ---------------------------------------------------------------------------


class TestLLMFactoryRouting:
    """测试 LLMFactory 对三种 OpenAI-compatible provider 的路由。"""

    def setup_method(self):
        """每个测试前清空工厂注册表，防止污染。"""
        LLMFactory._custom_registry.clear()

    def test_factory_routes_to_openai(self):
        with patch("openai.OpenAI"):
            LLMFactory.register("openai", OpenAILLM)
            llm = LLMFactory.create(make_settings(provider="openai"))
        assert isinstance(llm, OpenAILLM)
        assert llm.provider == "openai"

    def test_factory_routes_to_azure(self):
        with patch("openai.AzureOpenAI"):
            LLMFactory.register("azure", AzureLLM)
            llm = LLMFactory.create(make_settings(provider="azure"))
        assert isinstance(llm, AzureLLM)
        assert llm.provider == "azure"

    def test_factory_routes_to_deepseek(self):
        with patch("openai.OpenAI"):
            LLMFactory.register("deepseek", DeepSeekLLM)
            llm = LLMFactory.create(make_settings(provider="deepseek"))
        assert isinstance(llm, DeepSeekLLM)
        assert llm.provider == "deepseek"

    def test_factory_provider_case_insensitive(self):
        """provider 名称大小写不敏感。"""
        with patch("openai.OpenAI"):
            LLMFactory.register("OpenAI", OpenAILLM)  # 大写注册
            s = make_settings(provider="openai")       # 小写查询
            llm = LLMFactory.create(s)
        assert isinstance(llm, OpenAILLM)

    def test_factory_unknown_provider_raises_llm_error(self):
        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create(make_settings(provider="nonexistent"))
        err = str(exc_info.value)
        assert "nonexistent" in err

    def test_factory_error_contains_registered_list(self):
        LLMFactory.register("openai", OpenAILLM)
        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create(make_settings(provider="unknown"))
        assert "openai" in str(exc_info.value)

    def test_factory_all_three_registered_simultaneously(self):
        """同时注册三个 provider，按 settings 路由互不干扰。"""
        with patch("openai.OpenAI"), patch("openai.AzureOpenAI"):
            LLMFactory.register("openai", OpenAILLM)
            LLMFactory.register("azure", AzureLLM)
            LLMFactory.register("deepseek", DeepSeekLLM)

            llm_oa = LLMFactory.create(make_settings(provider="openai"))
            llm_az = LLMFactory.create(make_settings(provider="azure"))
            llm_ds = LLMFactory.create(make_settings(provider="deepseek"))

        assert isinstance(llm_oa, OpenAILLM)
        assert isinstance(llm_az, AzureLLM)
        assert isinstance(llm_ds, DeepSeekLLM)


# ---------------------------------------------------------------------------
# End-to-End Smoke: register + create + chat (via factory)
# ---------------------------------------------------------------------------


class TestE2ESmoke:
    """端到端 smoke 测试：通过工厂创建后调用 chat()。"""

    def setup_method(self):
        LLMFactory._custom_registry.clear()

    def test_openai_e2e(self):
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("smoke ok openai")
            MockClient.return_value.chat.completions.create.return_value = fake_resp

            LLMFactory.register("openai", OpenAILLM)
            llm = LLMFactory.create(make_settings(provider="openai"))
            result = llm.chat(make_messages("smoke test"))
        assert result == "smoke ok openai"

    def test_azure_e2e(self):
        with patch("openai.AzureOpenAI") as MockClient:
            fake_resp = make_fake_completion("smoke ok azure")
            MockClient.return_value.chat.completions.create.return_value = fake_resp

            LLMFactory.register("azure", AzureLLM)
            llm = LLMFactory.create(make_settings(provider="azure"))
            result = llm.chat(make_messages("smoke test"))
        assert result == "smoke ok azure"

    def test_deepseek_e2e(self):
        with patch("openai.OpenAI") as MockClient:
            fake_resp = make_fake_completion("smoke ok deepseek")
            MockClient.return_value.chat.completions.create.return_value = fake_resp

            LLMFactory.register("deepseek", DeepSeekLLM)
            llm = LLMFactory.create(make_settings(provider="deepseek"))
            result = llm.chat(make_messages("smoke test"))
        assert result == "smoke ok deepseek"
