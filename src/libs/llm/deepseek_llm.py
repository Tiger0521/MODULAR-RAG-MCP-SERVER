"""DeepSeek LLM 实现（provider="deepseek"）。

设计原则
--------
- DeepSeek API 与 OpenAI Chat Completions API 完全兼容，复用 ``openai.OpenAI`` 客户端。
- 默认 ``base_url`` 为 ``https://api.deepseek.com``，可通过 ``settings.llm.base_url`` 覆盖。
- 默认模型为 ``deepseek-chat``，可通过 ``settings.llm.model`` 覆盖。
- 接口契约与 ``OpenAILLM`` 完全相同（LLMError、空消息校验等）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from src.libs.llm.base_llm import BaseLLM, LLMError, Message

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.types import TraceContext

# DeepSeek 默认 API 端点
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekLLM(BaseLLM):
    """DeepSeek Chat LLM（OpenAI-Compatible API）。

    DeepSeek 提供与 OpenAI Chat Completions 完全兼容的 API，本实现通过
    ``openai.OpenAI`` 客户端并指定 ``base_url=https://api.deepseek.com`` 访问。

    Parameters
    ----------
    settings:
        全局 ``Settings`` 配置对象。读取：
        - ``settings.llm.api_key``  — DeepSeek API Key（可为 None，测试时注入 placeholder）。
        - ``settings.llm.model``    — 模型名称（默认 ``deepseek-chat``）。
        - ``settings.llm.base_url`` — 自定义 base URL（可覆盖 DeepSeek 默认端点）。
    """

    def __init__(self, settings: "Settings") -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError(
                "[deepseek] 缺少依赖：请安装 openai SDK（pip install openai）。"
            ) from exc

        self._settings = settings
        llm_cfg = settings.llm

        base_url = llm_cfg.base_url or _DEEPSEEK_BASE_URL

        try:
            self._client = openai.OpenAI(
                api_key=llm_cfg.api_key or "placeholder",
                base_url=base_url,
            )
        except Exception as exc:
            raise LLMError(f"[{self.provider}] 初始化 DeepSeek 客户端失败: {exc}") from exc

        self._model = llm_cfg.model or "deepseek-chat"

    @property
    def provider(self) -> str:
        return "deepseek"

    def chat(
        self,
        messages: List[Message],
        trace: Optional["TraceContext"] = None,
        **kwargs: Any,
    ) -> str:
        """调用 DeepSeek Chat API 并返回回复文本。

        Parameters
        ----------
        messages:
            对话消息列表；不能为 None，允许为空列表（将直接抛出 LLMError）。
        trace:
            可选追踪上下文（DeepSeekLLM 当前不打点，参数保留满足接口约定）。
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
                model=self._model,
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
                f"[{self.provider}] API Key 无效或未设置: {exc}"
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
