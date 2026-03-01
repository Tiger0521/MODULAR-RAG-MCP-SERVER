"""LLM 抽象基类、消息数据结构与自定义异常。

设计原则
--------
- Pluggable       — 抽象接口 + ``LLMFactory`` 按 provider 路由，上层无需感知具体实现。
- Config-Driven   — 具体实现构造函数接收 ``Settings``，provider 来自 settings.yaml。
- Observable      — ``chat()`` 接收可选 ``TraceContext``，供 F* 阶段打点使用。
- Graceful Error  — 统一使用 ``LLMError`` 汇报可读错误，包含 provider 名称与原因。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# 延迟导入避免循环依赖；运行时才需要 TraceContext
try:
    from src.core.types import TraceContext
except ImportError:  # pragma: no cover
    TraceContext = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """LLM 相关错误的统一基类。

    错误信息应包含 provider 名称与可读原因，便于排查配置或 API 问题。
    """


# ---------------------------------------------------------------------------
# 消息数据结构
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """单条对话消息（role + content）。

    Parameters
    ----------
    role:
        消息角色，标准值为 ``"system"``、``"user"``、``"assistant"``。
    content:
        消息文本内容。
    """

    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """将消息转换为字典表示（兼容 OpenAI SDK 格式）。"""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseLLM(ABC):
    """LLM 统一抽象接口。

    所有具体实现（OpenAI、Azure、Ollama、DeepSeek 等）必须继承此类并实现
    ``provider`` 属性与 ``chat`` 方法。

    Interface Contract
    ------------------
    - ``provider`` 属性：返回与 ``settings.llm.provider`` 匹配的小写字符串。
    - ``chat(messages, trace=None, **kwargs) -> str``：执行对话推理，返回纯字符串内容。
      如需完整响应元数据（usage、finish_reason 等），可在子类中扩展。

    Raises
    ------
    LLMError
        底层 API 调用失败或响应不符合预期时应统一转换并抛出 ``LLMError``。
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """返回 provider 标识符（与 ``settings.llm.provider`` 的值对应，小写）。"""

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        trace: Optional["TraceContext"] = None,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> str:
        """执行对话推理，返回模型回复内容（纯字符串）。

        Parameters
        ----------
        messages:
            对话历史，按 ``[system?, user, assistant?, user, ...]`` 顺序排列。
            允许传入空列表，具体实现可选择返回空字符串或抛出 ``LLMError``。
        trace:
            可选追踪上下文，由上层 pipeline 传入；实现类可在此打点。
        **kwargs:
            透传给底层 SDK 的额外参数（如 ``temperature``、``max_tokens``）。

        Returns
        -------
        str
            模型回复的文本内容。

        Raises
        ------
        LLMError
            底层 API 调用失败时抛出，错误信息包含 provider 名称与原始错误描述。
        """
