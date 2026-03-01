"""LLM 工厂：按配置的 provider 创建对应 BaseLLM 实例。

设计原则
--------
- Config-Driven  — provider 名称来自 ``settings.llm.provider``（settings.yaml）。
- Pluggable      — 通过注册表模式（``register``）接受自定义 provider，无需修改工厂本体。
- Graceful Error — 未知 provider 时抛出 ``LLMError``，错误信息包含 provider 名称与已注册列表。
"""

from __future__ import annotations

from typing import Dict, Type

from src.libs.llm.base_llm import BaseLLM, LLMError

# 延迟导入：Settings 定义在 core 层，避免循环依赖
try:
    from src.core.settings import Settings
except ImportError:  # pragma: no cover
    Settings = None  # type: ignore[assignment,misc]


class LLMFactory:
    """LLM 工厂。

    通过注册表模式将 provider 名称映射到具体实现类，工厂本身不硬编码 import
    路径，保持低耦合。

    Usage
    -----
    >>> LLMFactory.register("fake", FakeLLM)
    >>> llm = LLMFactory.create(settings)  # settings.llm.provider == "fake"

    Notes
    -----
    ``_custom_registry`` 为类级别字典，测试夹具可通过 ``clear()`` 重置，
    防止不同测试间互相污染。
    """

    # 存放用户注册的 provider → 实现类 的映射
    _custom_registry: Dict[str, Type[BaseLLM]] = {}

    # ------------------------------------------------------------------
    # 注册 API
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, provider_name: str, llm_class: Type[BaseLLM]) -> None:
        """注册一个 provider 的实现类。

        Parameters
        ----------
        provider_name:
            与 ``settings.llm.provider`` 匹配的字符串（统一转小写存储）。
        llm_class:
            ``BaseLLM`` 子类；工厂将以 ``llm_class(settings)`` 的方式实例化。
        """
        cls._custom_registry[provider_name.lower()] = llm_class

    @classmethod
    def registered_providers(cls) -> list[str]:
        """返回当前已注册的 provider 名称列表（来自 ``_custom_registry``）。"""
        return list(cls._custom_registry.keys())

    # ------------------------------------------------------------------
    # 创建 API
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, settings: "Settings") -> BaseLLM:
        """按 ``settings.llm.provider`` 创建并返回对应 ``BaseLLM`` 实例。

        Parameters
        ----------
        settings:
            全局 ``Settings`` 配置对象（``settings.llm.provider`` 决定路由）。

        Returns
        -------
        BaseLLM
            已初始化的 LLM 实例，由 ``llm_class(settings)`` 创建。

        Raises
        ------
        LLMError
            当 ``settings.llm.provider`` 未在注册表中找到时抛出，错误信息包含
            provider 名称与当前已注册列表，方便排查配置错误。
        """
        provider_key = settings.llm.provider.lower()

        llm_class = cls._custom_registry.get(provider_key)
        if llm_class is None:
            known = cls.registered_providers()
            raise LLMError(
                f"未知的 LLM provider: '{settings.llm.provider}'。"
                f"已注册的 providers: {known}。"
                f"请检查 config/settings.yaml 中的 llm.provider 字段。"
            )

        return llm_class(settings)

