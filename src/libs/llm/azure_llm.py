"""Azure OpenAI LLM 实现（provider="azure"）。

设计原则
--------
- 使用官方 ``openai`` SDK 的 ``openai.AzureOpenAI`` 客户端。
- 从 ``settings.llm`` 读取 Azure 特有字段：
  ``azure_api_key`` / ``azure_endpoint`` / ``azure_api_version`` / ``model``（作为 deployment_name）。
- 其余接口契约与 ``OpenAILLM`` 完全相同（LLMError、空消息校验等）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from src.libs.llm.base_llm import BaseLLM, LLMError, Message

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.types import TraceContext


class AzureLLM(BaseLLM):
    """Azure OpenAI Chat Completion LLM。

    通过 ``openai.AzureOpenAI`` 客户端调用 Azure OpenAI Services 的
    Chat Completions API（支持 gpt-4o、gpt-4-turbo 等部署模型）。

    Parameters
    ----------
    settings:
        全局 ``Settings`` 配置对象。读取：
        - ``settings.llm.azure_api_key``    — Azure OpenAI API Key。
        - ``settings.llm.azure_endpoint``   — Azure 端点 URL（如 ``https://<resource>.openai.azure.com/``）。
        - ``settings.llm.azure_api_version``— API 版本（默认 ``2024-02-15-preview``）。
        - ``settings.llm.model``            — 部署名称（deployment name）。
    """

    def __init__(self, settings: "Settings") -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError(
                "[azure] 缺少依赖：请安装 openai SDK（pip install openai）。"
            ) from exc

        self._settings = settings
        llm_cfg = settings.llm

        try:
            self._client = openai.AzureOpenAI(
                api_key=llm_cfg.azure_api_key or "placeholder",
                azure_endpoint=llm_cfg.azure_endpoint or "https://placeholder.openai.azure.com/",
                api_version=llm_cfg.azure_api_version or "2024-02-15-preview",
            )
        except Exception as exc:
            raise LLMError(f"[{self.provider}] 初始化 AzureOpenAI 客户端失败: {exc}") from exc

        # Azure 中 model 字段即 deployment name
        self._deployment = llm_cfg.model or "gpt-4o"

    @property
    def provider(self) -> str:
        return "azure"

    def chat(
        self,
        messages: List[Message],
        trace: Optional["TraceContext"] = None,
        **kwargs: Any,
    ) -> str:
        """调用 Azure OpenAI Chat Completions API 并返回回复文本。

        Parameters
        ----------
        messages:
            对话消息列表；不能为 None，允许为空列表（将直接抛出 LLMError）。
        trace:
            可选追踪上下文（AzureLLM 当前不打点，参数保留满足接口约定）。
        **kwargs:
            透传给 API 的额外参数（如 ``temperature``、``max_tokens``）。

        Returns
        -------
        str
            模型回复的文本内容。

        Raises
        ------
        LLMError
            messages 为 None、API 调用失败、响应格式异常时抛出。
        """
        if messages is None:
            raise LLMError(
                f"[{self.provider}] chat() 的 messages 参数不能为 None。"
            )
        if not messages:
            raise LLMError(
                f"[{self.provider}] chat() 的 messages 列表不能为空。"
            )

        try:
            import openai  # noqa: PLC0415

            api_messages = [m.to_dict() for m in messages]
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=api_messages,
                **kwargs,
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMError(
                    f"[{self.provider}] API 返回空内容（choices[0].message.content 为 None）。"
                )
            return content

        except LLMError:
            raise
        except openai.AuthenticationError as exc:
            raise LLMError(
                f"[{self.provider}] Azure API Key 无效或未设置: {exc}"
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMError(
                f"[{self.provider}] 超出速率限制 (Rate Limit): {exc}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMError(
                f"[{self.provider}] 网络连接失败: {exc}"
            ) from exc
        except openai.APITimeoutError as exc:
            raise LLMError(
                f"[{self.provider}] 请求超时: {exc}"
            ) from exc
        except openai.APIStatusError as exc:
            raise LLMError(
                f"[{self.provider}] API 返回错误状态码 {exc.status_code}: {exc.message}"
            ) from exc
        except Exception as exc:
            raise LLMError(
                f"[{self.provider}] 调用失败: {exc}"
            ) from exc
