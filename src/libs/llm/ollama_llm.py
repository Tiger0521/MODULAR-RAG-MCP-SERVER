"""Ollama LLM 实现（provider="ollama"）。

设计原则
--------
- Ollama 提供与 OpenAI Chat Completions 兼容的 REST API（/v1/chat/completions），
  本实现通过 ``openai.OpenAI`` 客户端连接本地 Ollama 服务。
- 默认 base_url 为 ``http://localhost:11434/v1``（来自 settings.llm.ollama_base_url）。
  可通过 ``settings.llm.base_url`` 覆盖（适用于远程/代理 Ollama 实例）。
- 默认模型为 ``llama3.1:8b``（来自 settings.llm.ollama_model），本地模型无需 API Key。
- 连接失败、超时等场景下抛出可读 LLMError，不泄露敏感配置信息。
- 延迟导入 openai SDK：允许在未安装 SDK 时 import 本模块，仅在实际使用时报错。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from src.libs.llm.base_llm import BaseLLM, LLMError, Message

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.types import TraceContext

# Ollama OpenAI-compatible API 路径后缀
_OLLAMA_API_SUFFIX = "/v1"


class OllamaLLM(BaseLLM):
    """Ollama 本地 LLM（OpenAI-Compatible API）。

    Ollama 在本地暴露与 OpenAI Chat Completions 兼容的 REST 接口，本实现通过
    ``openai.OpenAI`` 客户端并指定 ``base_url`` 指向 Ollama 服务的 ``/v1`` 路径
    进行调用，无需真实 API Key。

    Parameters
    ----------
    settings:
        全局 ``Settings`` 配置对象。读取：
        - ``settings.llm.base_url``      — 自定义 base URL（可覆盖 Ollama 默认端点）。
        - ``settings.llm.ollama_base_url`` — Ollama 服务地址（默认 ``http://localhost:11434``）。
        - ``settings.llm.ollama_model``  — 模型名称（默认 ``llama3.1:8b``）。
    """

    def __init__(self, settings: "Settings") -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError(
                "[ollama] 缺少依赖：请安装 openai SDK（pip install openai）。"
            ) from exc

        self._settings = settings
        llm_cfg = settings.llm

        # base_url 优先级：显式 base_url > ollama_base_url（含默认 localhost:11434）
        raw_base = llm_cfg.base_url or llm_cfg.ollama_base_url
        # 确保路径包含 /v1 后缀（Ollama OpenAI-compatible endpoint）
        if raw_base.endswith("/v1"):
            base_url = raw_base
        elif raw_base.endswith("/"):
            base_url = raw_base + "v1"
        else:
            base_url = raw_base + _OLLAMA_API_SUFFIX

        try:
            self._client = openai.OpenAI(
                api_key="ollama",  # Ollama 不需要真实 API Key，传占位符满足 SDK 要求
                base_url=base_url,
            )
        except Exception as exc:
            raise LLMError(
                f"[{self.provider}] 初始化 Ollama 客户端失败: {exc}"
            ) from exc

        # 模型名称：使用 Ollama 专属字段（默认 llama3.1:8b）
        self._model = llm_cfg.ollama_model or "llama3.1:8b"
        self._base_url = base_url

    @property
    def provider(self) -> str:
        return "ollama"

    def chat(
        self,
        messages: List[Message],
        trace: Optional["TraceContext"] = None,
        **kwargs: Any,
    ) -> str:
        """调用 Ollama Chat API 并返回回复文本。

        Parameters
        ----------
        messages:
            对话消息列表；不能为 None，允许为空列表（将直接抛出 LLMError）。
        trace:
            可选追踪上下文（OllamaLLM 当前不打点，参数保留满足接口约定）。
        **kwargs:
            透传给 API 的额外参数（如 ``temperature``、``max_tokens``）。

        Returns
        -------
        str
            模型回复的文本内容。

        Raises
        ------
        LLMError
            messages 为 None、API 调用失败（连接失败/超时/模型不存在等）时抛出。
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
        except openai.APITimeoutError as exc:
            # 注意：APITimeoutError 是 APIConnectionError 的子类，必须先捕获
            raise LLMError(
                f"[{self.provider}] 请求超时：Ollama 服务响应过慢，请检查服务状态或调整超时配置。"
                f" 错误: {exc}"
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMError(
                f"[{self.provider}] 无法连接 Ollama 服务（{self._base_url}）："
                f"请确认 Ollama 已启动并监听该地址。错误: {exc}"
            ) from exc
        except openai.AuthenticationError as exc:
            raise LLMError(
                f"[{self.provider}] 认证失败（通常不应出现于本地 Ollama）: {exc}"
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMError(
                f"[{self.provider}] 超出速率限制: {exc}"
            ) from exc
        except openai.NotFoundError as exc:
            raise LLMError(
                f"[{self.provider}] 模型 '{self._model}' 未找到："
                f"请确认已通过 `ollama pull {self._model}` 下载该模型。错误: {exc}"
            ) from exc
        except openai.APIStatusError as exc:
            raise LLMError(
                f"[{self.provider}] API 返回错误状态码 {exc.status_code}: {exc.message}"
            ) from exc
        except Exception as exc:
            raise LLMError(
                f"[{self.provider}] 调用失败: {exc}"
            ) from exc
