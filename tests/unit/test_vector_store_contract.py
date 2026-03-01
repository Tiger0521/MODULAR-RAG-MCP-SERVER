"""Unit tests for B4: VectorStore 抽象接口与工厂（契约测试）

验收标准
--------
- VectorRecord / QueryResult 数据契约 shape 正确（字段类型、默认值、校验）。
- BaseVectorStore 不可直接实例化（抽象类）。
- FakeVectorStore 满足完整接口契约。
- VectorStoreFactory 按 provider 正确分流。
- 未知 provider 时 VectorStoreError 消息包含 provider 名称与已注册列表。
- upsert() 空列表抛出 ValueError；query() top_k < 1 抛出 ValueError。
- delete() / delete_by_metadata() 空入参抛出 ValueError。
- upsert/query 支持可选 TraceContext 参数。
- 幂等语义：相同 id 二次 upsert 覆盖而非追加。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    QueryResult,
    VectorRecord,
    VectorStoreError,
)
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.core.settings import Settings


# ---------------------------------------------------------------------------
# Fake VectorStore（行内 Stub，不依赖任何外部 DB）
# ---------------------------------------------------------------------------


class FakeVectorStore(BaseVectorStore):
    """确定性 Fake VectorStore：内存字典存储，结果稳定可重现。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # id → VectorRecord
        self._store: Dict[str, VectorRecord] = {}
        # 调用计数（用于验证调用链路）
        self.upsert_calls: int = 0
        self.query_calls: int = 0

    @property
    def provider(self) -> str:
        return "fake"

    def upsert(
        self,
        records: List[VectorRecord],
        trace=None,
        **kwargs: Any,
    ) -> int:
        if not records:
            raise ValueError("records 不能为空列表")
        self.upsert_calls += 1
        for rec in records:
            self._store[rec.id] = rec
        return len(records)

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace=None,
        **kwargs: Any,
    ) -> List[QueryResult]:
        if not vector:
            raise ValueError("query vector 不能为空")
        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")
        self.query_calls += 1

        # 简单内积相似度（确定性，不依赖 numpy）
        def dot(a: List[float], b: List[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        candidates = list(self._store.values())

        # 应用 metadata 过滤
        if filters:
            candidates = [
                r for r in candidates
                if all(r.metadata.get(k) == v for k, v in filters.items())
            ]

        scored = [
            QueryResult(
                id=r.id,
                score=dot(vector, r.vector),
                text=r.text,
                metadata=r.metadata,
            )
            for r in candidates
            if len(r.vector) == len(vector)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def delete(
        self,
        ids: List[str],
        trace=None,
        **kwargs: Any,
    ) -> int:
        if not ids:
            raise ValueError("ids 不能为空列表")
        deleted = 0
        for id_ in ids:
            if id_ in self._store:
                del self._store[id_]
                deleted += 1
        return deleted

    def delete_by_metadata(
        self,
        filter: Dict[str, Any],
        trace=None,
        **kwargs: Any,
    ) -> int:
        if not filter:
            raise ValueError("filter 不能为空 dict，防止误删全库")
        to_delete = [
            id_
            for id_, rec in self._store.items()
            if all(rec.metadata.get(k) == v for k, v in filter.items())
        ]
        for id_ in to_delete:
            del self._store[id_]
        return len(to_delete)


class AnotherFakeVectorStore(BaseVectorStore):
    """另一个 Fake，用于验证多 provider 路由。"""

    def __init__(self, settings: Settings) -> None:
        pass

    @property
    def provider(self) -> str:
        return "another_fake"

    def upsert(self, records, trace=None, **kwargs):
        if not records:
            raise ValueError("records 不能为空")
        return len(records)

    def query(self, vector, top_k, filters=None, trace=None, **kwargs):
        return []

    def delete(self, ids, trace=None, **kwargs):
        if not ids:
            raise ValueError("ids 不能为空")
        return 0

    def delete_by_metadata(self, filter, trace=None, **kwargs):
        if not filter:
            raise ValueError("filter 不能为空 dict")
        return 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试前后清理注册表，防止污染。"""
    VectorStoreFactory._custom_registry.clear()
    yield
    VectorStoreFactory._custom_registry.clear()


@pytest.fixture()
def fake_settings() -> Settings:
    s = Settings()
    s.vector_store.provider = "fake"
    return s


@pytest.fixture()
def unknown_settings() -> Settings:
    s = Settings()
    s.vector_store.provider = "unknown_provider_xyz"
    return s


@pytest.fixture()
def fake_store(fake_settings) -> FakeVectorStore:
    VectorStoreFactory.register("fake", FakeVectorStore)
    return VectorStoreFactory.create(fake_settings)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# VectorRecord 契约测试
# ---------------------------------------------------------------------------


class TestVectorRecordContract:
    def test_create_valid_record(self):
        rec = VectorRecord(id="doc-1", vector=[0.1, 0.2, 0.3], text="hello")
        assert rec.id == "doc-1"
        assert rec.vector == [0.1, 0.2, 0.3]
        assert rec.text == "hello"
        assert rec.metadata == {}

    def test_record_with_metadata(self):
        rec = VectorRecord(
            id="doc-2",
            vector=[1.0, 0.0],
            text="world",
            metadata={"source": "test.pdf", "page": 1},
        )
        assert rec.metadata["source"] == "test.pdf"
        assert rec.metadata["page"] == 1

    def test_empty_id_raises_value_error(self):
        with pytest.raises(ValueError, match="id"):
            VectorRecord(id="", vector=[0.1], text="test")

    def test_empty_vector_raises_value_error(self):
        with pytest.raises(ValueError, match="vector"):
            VectorRecord(id="doc-1", vector=[], text="test")


# ---------------------------------------------------------------------------
# QueryResult 契约测试
# ---------------------------------------------------------------------------


class TestQueryResultContract:
    def test_create_valid_result(self):
        res = QueryResult(id="doc-1", score=0.95, text="hello")
        assert res.id == "doc-1"
        assert res.score == 0.95
        assert res.metadata == {}

    def test_create_result_with_metadata(self):
        res = QueryResult(
            id="doc-2",
            score=0.8,
            text="world",
            metadata={"page": 3},
        )
        assert res.metadata["page"] == 3


# ---------------------------------------------------------------------------
# BaseVectorStore 接口约束
# ---------------------------------------------------------------------------


class TestBaseVectorStoreContract:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseVectorStore()  # type: ignore[abstract]

    def test_fake_store_satisfies_interface(self, fake_settings):
        store = FakeVectorStore(fake_settings)
        assert isinstance(store, BaseVectorStore)

    def test_provider_returns_correct_name(self, fake_settings):
        store = FakeVectorStore(fake_settings)
        assert store.provider == "fake"


# ---------------------------------------------------------------------------
# VectorStoreFactory 路由测试
# ---------------------------------------------------------------------------


class TestVectorStoreFactory:
    def test_register_and_create_fake_provider(self, fake_settings):
        VectorStoreFactory.register("fake", FakeVectorStore)
        store = VectorStoreFactory.create(fake_settings)
        assert isinstance(store, FakeVectorStore)

    def test_create_returns_base_vector_store(self, fake_settings):
        VectorStoreFactory.register("fake", FakeVectorStore)
        store = VectorStoreFactory.create(fake_settings)
        assert isinstance(store, BaseVectorStore)

    def test_factory_routes_to_correct_class(self, fake_settings):
        VectorStoreFactory.register("fake", FakeVectorStore)
        VectorStoreFactory.register("another_fake", AnotherFakeVectorStore)

        fake_settings.vector_store.provider = "another_fake"
        store = VectorStoreFactory.create(fake_settings)
        assert isinstance(store, AnotherFakeVectorStore)

    def test_unknown_provider_raises_vector_store_error(self, unknown_settings):
        with pytest.raises(VectorStoreError) as exc_info:
            VectorStoreFactory.create(unknown_settings)
        assert "unknown_provider_xyz" in str(exc_info.value)

    def test_error_message_lists_registered_providers(self, fake_settings):
        VectorStoreFactory.register("fake", FakeVectorStore)
        fake_settings.vector_store.provider = "not_exist"
        with pytest.raises(VectorStoreError) as exc_info:
            VectorStoreFactory.create(fake_settings)
        assert "fake" in str(exc_info.value)

    def test_provider_name_case_insensitive(self, fake_settings):
        VectorStoreFactory.register("FAKE_UPPER", FakeVectorStore)
        fake_settings.vector_store.provider = "fake_upper"
        store = VectorStoreFactory.create(fake_settings)
        assert isinstance(store, FakeVectorStore)
        VectorStoreFactory._custom_registry.pop("fake_upper", None)

    def test_registered_providers_returns_list(self, fake_settings):
        VectorStoreFactory.register("fake", FakeVectorStore)
        providers = VectorStoreFactory.registered_providers()
        assert isinstance(providers, list)
        assert "fake" in providers

    def test_register_overwrites_existing_provider(self, fake_settings):
        VectorStoreFactory.register("fake", FakeVectorStore)
        VectorStoreFactory.register("fake", AnotherFakeVectorStore)
        store = VectorStoreFactory.create(fake_settings)
        assert isinstance(store, AnotherFakeVectorStore)

    def test_empty_registry_raises_vector_store_error(self, fake_settings):
        with pytest.raises(VectorStoreError):
            VectorStoreFactory.create(fake_settings)


# ---------------------------------------------------------------------------
# BaseVectorStore.upsert() 行为测试
# ---------------------------------------------------------------------------


class TestUpsertContract:
    def test_upsert_returns_count(self, fake_store):
        records = [
            VectorRecord(id="a", vector=[1.0, 0.0], text="doc a"),
            VectorRecord(id="b", vector=[0.0, 1.0], text="doc b"),
        ]
        count = fake_store.upsert(records)
        assert count == 2

    def test_upsert_empty_raises_value_error(self, fake_store):
        with pytest.raises(ValueError):
            fake_store.upsert([])

    def test_upsert_idempotent_same_id(self, fake_store):
        """相同 id 二次 upsert 应覆盖，库中记录数不增加。"""
        rec1 = VectorRecord(id="x", vector=[1.0, 0.0], text="v1")
        rec2 = VectorRecord(id="x", vector=[0.5, 0.5], text="v2")
        fake_store.upsert([rec1])
        fake_store.upsert([rec2])
        # 相同 id 只保留最新
        assert len(fake_store._store) == 1
        assert fake_store._store["x"].text == "v2"

    def test_upsert_accepts_trace_context(self, fake_store):
        rec = VectorRecord(id="trace-test", vector=[1.0], text="trace")
        # trace=None 不应抛出
        count = fake_store.upsert([rec], trace=None)
        assert count == 1

    def test_upsert_increments_call_count(self, fake_store):
        fake_store.upsert([VectorRecord(id="c1", vector=[1.0], text="t")])
        fake_store.upsert([VectorRecord(id="c2", vector=[2.0], text="t")])
        assert fake_store.upsert_calls == 2


# ---------------------------------------------------------------------------
# BaseVectorStore.query() 行为测试
# ---------------------------------------------------------------------------


class TestQueryContract:
    def _seed(self, store: FakeVectorStore) -> None:
        store.upsert([
            VectorRecord(id="a", vector=[1.0, 0.0], text="doc a", metadata={"cat": "x"}),
            VectorRecord(id="b", vector=[0.0, 1.0], text="doc b", metadata={"cat": "y"}),
            VectorRecord(id="c", vector=[0.7, 0.7], text="doc c", metadata={"cat": "x"}),
        ])

    def test_query_returns_list_of_query_results(self, fake_store):
        self._seed(fake_store)
        results = fake_store.query(vector=[1.0, 0.0], top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, QueryResult) for r in results)

    def test_query_top_k_limits_results(self, fake_store):
        self._seed(fake_store)
        results = fake_store.query(vector=[1.0, 0.0], top_k=2)
        assert len(results) <= 2

    def test_query_results_sorted_by_score_desc(self, fake_store):
        self._seed(fake_store)
        results = fake_store.query(vector=[1.0, 0.0], top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_with_metadata_filter(self, fake_store):
        self._seed(fake_store)
        results = fake_store.query(vector=[1.0, 0.0], top_k=3, filters={"cat": "x"})
        assert all(r.metadata.get("cat") == "x" for r in results)
        assert len(results) == 2  # a + c

    def test_query_empty_vector_raises_value_error(self, fake_store):
        with pytest.raises(ValueError):
            fake_store.query(vector=[], top_k=5)

    def test_query_top_k_zero_raises_value_error(self, fake_store):
        with pytest.raises(ValueError):
            fake_store.query(vector=[1.0, 0.0], top_k=0)

    def test_query_top_k_negative_raises_value_error(self, fake_store):
        with pytest.raises(ValueError):
            fake_store.query(vector=[1.0, 0.0], top_k=-1)

    def test_query_result_has_correct_fields(self, fake_store):
        self._seed(fake_store)
        results = fake_store.query(vector=[1.0, 0.0], top_k=1)
        assert len(results) == 1
        r = results[0]
        assert r.id == "a"
        assert isinstance(r.score, float)
        assert isinstance(r.text, str)
        assert isinstance(r.metadata, dict)

    def test_query_accepts_trace_context(self, fake_store):
        self._seed(fake_store)
        results = fake_store.query(vector=[1.0, 0.0], top_k=1, trace=None)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# BaseVectorStore.delete() 行为测试
# ---------------------------------------------------------------------------


class TestDeleteContract:
    def test_delete_existing_records(self, fake_store):
        fake_store.upsert([
            VectorRecord(id="d1", vector=[1.0], text="t1"),
            VectorRecord(id="d2", vector=[2.0], text="t2"),
        ])
        deleted = fake_store.delete(["d1", "d2"])
        assert deleted == 2
        assert len(fake_store._store) == 0

    def test_delete_nonexistent_id_returns_zero(self, fake_store):
        deleted = fake_store.delete(["not-exist"])
        assert deleted == 0

    def test_delete_empty_ids_raises_value_error(self, fake_store):
        with pytest.raises(ValueError):
            fake_store.delete([])

    def test_delete_partial_match(self, fake_store):
        fake_store.upsert([VectorRecord(id="e1", vector=[1.0], text="t")])
        deleted = fake_store.delete(["e1", "not-exist"])
        assert deleted == 1


# ---------------------------------------------------------------------------
# BaseVectorStore.delete_by_metadata() 行为测试
# ---------------------------------------------------------------------------


class TestDeleteByMetadataContract:
    def test_delete_by_metadata_matching_records(self, fake_store):
        fake_store.upsert([
            VectorRecord(id="f1", vector=[1.0], text="t1", metadata={"source": "a.pdf"}),
            VectorRecord(id="f2", vector=[2.0], text="t2", metadata={"source": "b.pdf"}),
            VectorRecord(id="f3", vector=[3.0], text="t3", metadata={"source": "a.pdf"}),
        ])
        deleted = fake_store.delete_by_metadata({"source": "a.pdf"})
        assert deleted == 2
        assert "f1" not in fake_store._store
        assert "f3" not in fake_store._store
        assert "f2" in fake_store._store

    def test_delete_by_metadata_no_match_returns_zero(self, fake_store):
        fake_store.upsert([VectorRecord(id="g1", vector=[1.0], text="t")])
        deleted = fake_store.delete_by_metadata({"source": "notexist.pdf"})
        assert deleted == 0

    def test_delete_by_metadata_empty_filter_raises_value_error(self, fake_store):
        """空 filter 应拒绝执行，防止误删全库。"""
        with pytest.raises(ValueError):
            fake_store.delete_by_metadata({})


# ---------------------------------------------------------------------------
# 全场景集成：upsert → query → delete 链路
# ---------------------------------------------------------------------------


class TestFullRoundtrip:
    def test_upsert_query_delete_roundtrip(self, fake_store):
        # 1. upsert
        records = [
            VectorRecord(id="r1", vector=[1.0, 0.0], text="alpha", metadata={"tag": "A"}),
            VectorRecord(id="r2", vector=[0.0, 1.0], text="beta", metadata={"tag": "B"}),
        ]
        count = fake_store.upsert(records)
        assert count == 2

        # 2. query
        results = fake_store.query(vector=[1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].id == "r1"  # 最高分

        # 3. delete
        deleted = fake_store.delete(["r1"])
        assert deleted == 1

        # 4. query again — r1 already gone
        results_after = fake_store.query(vector=[1.0, 0.0], top_k=2)
        ids = [r.id for r in results_after]
        assert "r1" not in ids
        assert "r2" in ids
