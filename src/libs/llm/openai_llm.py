"""OpenAI-Compatible LLM 实现（provider="openai"）。

设计原则
--------
- 使用官方 ``openai`` SDK，通过 ``openai.OpenAI`` 客户端发送请求。
- 构造函数接收 ``Settings``：从 ``settings.llm`` 读取 api_key / model / base_url。
- chat(messages) 接受空列表或 None 消息列表，抛出包含 provider 名称的 LLMError。
- 所有 OpenAI API 异常统一转换为 LLMError，错误信息包含 provider 与原始描述。
- 延迟导入 openai：允许在未安装 SDK 时 import 本模块，仅在实际使用时才报错。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from src.libs.llm.base_llm import BaseLLM, LLMError, Message

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.types import TraceContext


class OpenAILLM(BaseLLM):
    """OpenAI Chat Completion LLM。

    通过 ``openai.OpenAI`` 客户端调用 Chat Completions API。
    支持 GPT-4o、GPT-4-Turbo、GPT-3.5-Turbo 等 OpenAI 官方模型。

    Parameters
    ----------
    settings:
        全局 ``Settings`` 配置对象。读取：
        - ``settings.llm.api_key``  — OpenAI API Key（可为 None，测试时可注入）。
        - ``settings.llm.model``    — 模型名称（默认 ``gpt-4o``）。
        - ``settings.llm.base_url`` — 自定义 base URL（可选，用于代理或兼容接口）。
    """

    def __init__(self, settings: "Settings") -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError(
                "[openai] 缺少依赖：请安装 openai SDK（pip install openai）。"
            ) from exc

        self._settings = settings
        llm_cfg = settings.llm

        client_kwargs: dict = {
            "api_key": llm_cfg.api_key or "placeholder",  # SDK 要求非空，测试注入 placeholder
        }
        if llm_cfg.base_url:
            client_kwargs["base_url"] = llm_cfg.base_url

        try:
            self._client = openai.OpenAI(**client_kwargs)
        except Exception as exc:
            raise LLMError(f"[{self.provider}] 初始化 OpenAI 客户端失败: {exc}") from exc

        self._model = llm_cfg.model or "gpt-4o"

    @property
    def provider(self) -> str:
        return "openai"

    def chat(
        self,
        messages: List[Message],
        trace: Optional["TraceContext"] = None,
        **kwargs: Any,
    ) -> str:
        """调用 OpenAI Chat Completions API 并返回回复文本。

        Parameters
        ----------
        messages:
            对话消息列表；不能为 None，允许为空列表（将直接抛出 LLMError）。
        trace:
            可选追踪上下文（OpenAILLM 当前不打点，参数保留满足接口约定）。
        **kwargs:
            透传给 ``client.chat.completions.create()`` 的额外参数
            （如 ``temperature``、``max_tokens``）。

        Returns
        -------
        str
            模型回复的文本内容（``choices[0].message.content``）。

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
