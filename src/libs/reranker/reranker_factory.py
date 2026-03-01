"""Reranker 工厂：按配置的 backend 创建对应 BaseReranker 实例。

设计原则
--------
- Config-Driven  — backend 名称来自 ``settings.reranker.backend``（settings.yaml）。
- Pluggable      — 通过注册表模式（``register``）接受自定义 backend，无需修改工厂本体。
- Graceful Error — 未知 backend 时抛出 ``RerankerError``，包含 backend 名称与已注册列表。
- None Fallback  — ``NoneReranker`` 作为内置默认 backend（"none"），无需手动注册。
"""

from __future__ import annotations

from typing import Dict, Type

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankerError

# 延迟导入避免循环依赖
try:
    from src.core.settings import Settings
except ImportError:  # pragma: no cover
    Settings = None  # type: ignore[assignment,misc]


class RerankerFactory:
    """Reranker 工厂。

    通过注册表模式将 backend 名称映射到具体实现类，工厂本身不硬编码 import
    路径，保持低耦合。内置 ``NoneReranker`` 作为 "none" backend 的默认实现。

    Usage
    -----
    >>> RerankerFactory.register("fake", FakeReranker)
    >>> reranker = RerankerFactory.create(settings)  # settings.reranker.backend == "fake"

    Notes
    -----
    ``_custom_registry`` 为类级别字典，测试夹具可通过 ``clear()`` 重置，
    防止不同测试间互相污染。
    内置的 "none" backend（``NoneReranker``）始终可用，无需注册。
    """

    # 存放用户注册的 backend → 实现类 的映射
    _custom_registry: Dict[str, Type[BaseReranker]] = {}

    # ------------------------------------------------------------------
    # 注册 API
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, backend_name: str, reranker_class: Type[BaseReranker]) -> None:
        """注册一个 backend 的实现类。

        Parameters
        ----------
        backend_name:
            与 ``settings.reranker.backend`` 匹配的字符串（统一转小写存储）。
        reranker_class:
            ``BaseReranker`` 子类；工厂将以 ``reranker_class(settings)`` 的方式实例化。
        """
        cls._custom_registry[backend_name.lower()] = reranker_class

    @classmethod
    def registered_backends(cls) -> list[str]:
        """返回当前已注册（自定义）的 backend 名称列表。

        注意：内置的 "none" backend 不在此列表中，但始终可用。
        """
        return list(cls._custom_registry.keys())

    # ------------------------------------------------------------------
    # 创建 API
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, settings: "Settings") -> BaseReranker:  # type: ignore[type-arg]
        """按 ``settings.reranker.backend`` 创建并返回对应 ``BaseReranker`` 实例。

        内置规则
        --------
        - ``backend="none"`` 始终返回 ``NoneReranker(settings)``，无需注册。
        - 其他 backend 需提前通过 ``register()`` 注册，否则抛出 ``RerankerError``。

        Parameters
        ----------
        settings:
            全局 ``Settings`` 配置对象（``settings.reranker.backend`` 决定路由）。

        Returns
        -------
        BaseReranker
            已初始化的 Reranker 实例。

        Raises
        ------
        RerankerError
            当 ``settings.reranker.backend`` 未在注册表中找到（且不是内置 "none"）时
            抛出，错误信息包含 backend 名称与当前已注册列表，方便排查配置错误。
        """
        backend_key = settings.reranker.backend.lower()

        # 内置 none 回退：无需注册，始终可用
        if backend_key == "none":
            return NoneReranker(settings)

        # 查找自定义注册表
        reranker_class = cls._custom_registry.get(backend_key)
        if reranker_class is None:
            known = ["none"] + cls.registered_backends()
            raise RerankerError(
                f"未知的 Reranker backend: '{settings.reranker.backend}'。"
                f"已注册的 backends: {known}。"
                f"请检查 config/settings.yaml 中的 reranker.backend 字段。"
            )

        return reranker_class(settings)
