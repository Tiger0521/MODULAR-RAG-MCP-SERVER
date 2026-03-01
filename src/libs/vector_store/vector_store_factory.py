"""VectorStore 工厂：按配置的 provider 创建对应 BaseVectorStore 实例。

设计原则
--------
- Config-Driven  — provider 名称来自 ``settings.vector_store.provider``（settings.yaml）。
- Pluggable      — 通过注册表模式（``register``）接受自定义 provider，无需修改工厂本体。
- Graceful Error — 未知 provider 时抛出 ``VectorStoreError``，包含 provider 名称与已注册列表。
"""

from __future__ import annotations

from typing import Dict, Type

from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError

# 延迟导入避免循环依赖
try:
    from src.core.settings import Settings
except ImportError:  # pragma: no cover
    Settings = None  # type: ignore[assignment,misc]


class VectorStoreFactory:
    """VectorStore 工厂。

    通过注册表模式将 provider 名称映射到具体实现类，工厂本身不硬编码 import
    路径，保持低耦合。

    Usage
    -----
    >>> VectorStoreFactory.register("fake", FakeVectorStore)
    >>> store = VectorStoreFactory.create(settings)  # settings.vector_store.provider == "fake"

    Notes
    -----
    ``_custom_registry`` 为类级别字典，测试夹具可通过 ``clear()`` 重置，
    防止不同测试间互相污染。
    """

    # 存放用户注册的 provider → 实现类 的映射
    _custom_registry: Dict[str, Type[BaseVectorStore]] = {}

    # ------------------------------------------------------------------
    # 注册 API
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, provider_name: str, store_class: Type[BaseVectorStore]) -> None:
        """注册一个 provider 的实现类。

        Parameters
        ----------
        provider_name:
            与 ``settings.vector_store.provider`` 匹配的字符串（统一转小写存储）。
        store_class:
            ``BaseVectorStore`` 子类；工厂将以 ``store_class(settings)`` 的方式实例化。
        """
        cls._custom_registry[provider_name.lower()] = store_class

    @classmethod
    def registered_providers(cls) -> list[str]:
        """返回当前已注册的 provider 名称列表（来自 ``_custom_registry``）。"""
        return list(cls._custom_registry.keys())

    # ------------------------------------------------------------------
    # 创建 API
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, settings: "Settings") -> BaseVectorStore:
        """按 ``settings.vector_store.provider`` 创建并返回对应 ``BaseVectorStore`` 实例。

        Parameters
        ----------
        settings:
            全局 ``Settings`` 配置对象（``settings.vector_store.provider`` 决定路由）。

        Returns
        -------
        BaseVectorStore
            已初始化的 VectorStore 实例，由 ``store_class(settings)`` 创建。

        Raises
        ------
        VectorStoreError
            当 ``settings.vector_store.provider`` 未在注册表中找到时抛出，错误信息包含
            provider 名称与当前已注册列表，方便排查配置错误。
        """
        provider_key = settings.vector_store.provider.lower()

        store_class = cls._custom_registry.get(provider_key)
        if store_class is None:
            known = cls.registered_providers()
            raise VectorStoreError(
                f"未知的 VectorStore provider: '{settings.vector_store.provider}'。"
                f"已注册的 providers: {known}。"
                f"请检查 config/settings.yaml 中的 vector_store.provider 字段。"
            )

        return store_class(settings)
