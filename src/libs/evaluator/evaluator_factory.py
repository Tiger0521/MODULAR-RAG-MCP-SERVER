"""Evaluator 工厂：按配置的 backend 创建对应 BaseEvaluator 实例。

设计原则
--------
- Config-Driven  — backend 名称来自 ``settings.evaluator.backend``（settings.yaml）。
- Pluggable      — 通过注册表模式（``register``）接受自定义 backend，无需修改工厂本体。
- Graceful Error — 未知 backend 时抛出 ``EvaluatorError``，包含 backend 名称与已注册列表。
"""

from __future__ import annotations

from typing import Dict, Type

from src.libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError

# 延迟导入避免循环依赖
try:
    from src.core.settings import Settings
except ImportError:  # pragma: no cover
    Settings = None  # type: ignore[assignment,misc]


class EvaluatorFactory:
    """Evaluator 工厂。

    通过注册表模式将 backend 名称映射到具体实现类，工厂本身不硬编码 import
    路径，保持低耦合。

    Usage
    -----
    >>> EvaluatorFactory.register("custom", CustomEvaluator)
    >>> evaluator = EvaluatorFactory.create(settings)  # settings.evaluator.backend == "custom"

    Notes
    -----
    ``_registry`` 为类级别字典，测试夹具可通过 ``clear()`` 重置，
    防止不同测试间互相污染。
    """

    # 存放 backend → 实现类 的映射
    _registry: Dict[str, Type[BaseEvaluator]] = {}

    # ------------------------------------------------------------------
    # 注册 API
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, backend_name: str, evaluator_class: Type[BaseEvaluator]) -> None:
        """注册一个 backend 的实现类。

        Parameters
        ----------
        backend_name:
            与 ``settings.evaluator.backend`` 匹配的字符串（统一转小写存储）。
        evaluator_class:
            ``BaseEvaluator`` 子类；工厂将以 ``evaluator_class(settings)`` 的方式实例化。
        """
        cls._registry[backend_name.lower()] = evaluator_class

    @classmethod
    def registered_backends(cls) -> list[str]:
        """返回当前已注册的 backend 名称列表。"""
        return list(cls._registry.keys())

    # ------------------------------------------------------------------
    # 创建 API
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, settings: "Settings") -> BaseEvaluator:  # type: ignore[type-arg]
        """按 ``settings.evaluator.backend`` 创建并返回对应 ``BaseEvaluator`` 实例。

        Parameters
        ----------
        settings:
            全局 ``Settings`` 配置对象（``settings.evaluator.backend`` 决定路由）。

        Returns
        -------
        BaseEvaluator
            已初始化的 Evaluator 实例。

        Raises
        ------
        EvaluatorError
            当 ``settings.evaluator.backend`` 未在注册表中找到时抛出，
            错误信息包含 backend 名称与当前已注册列表，方便排查配置错误。
        """
        backend_key = settings.evaluator.backend.lower()

        evaluator_class = cls._registry.get(backend_key)
        if evaluator_class is None:
            known = cls.registered_backends()
            raise EvaluatorError(
                f"未知的 Evaluator backend: '{settings.evaluator.backend}'。"
                f"已注册的 backends: {known}。"
                f"请检查 config/settings.yaml 中的 evaluator.backend 字段，"
                f"或先调用 EvaluatorFactory.register() 注册该 backend。"
            )

        return evaluator_class(settings)
