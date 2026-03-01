"""核心数据类型测试"""

import pytest
from datetime import datetime
from src.core.types import (
    Document, DocumentType, Chunk, ChunkType, ChunkRecord,
    QueryResult, SearchRequest, SearchResponse, IngestionProgress,
    TraceContext, ErrorResponse
)


class TestCoreTypes:
    """核心数据类型测试类"""
    
    def test_document_creation(self):
        """测试文档创建"""
        doc = Document(
            id="doc-123",
            title="测试文档",
            content="这是测试文档内容",
            type=DocumentType.TEXT
        )
        
        assert doc.id == "doc-123"
        assert doc.title == "测试文档"
        assert doc.content == "这是测试文档内容"
        assert doc.type == DocumentType.TEXT
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.updated_at, datetime)
    
    def test_chunk_creation(self):
        """测试分块创建"""
        chunk = Chunk(
            id="chunk-456",
            document_id="doc-123",
            content="这是分块内容",
            start_index=0,
            end_index=100,
            chunk_index=1
        )
        
        assert chunk.id == "chunk-456"
        assert chunk.document_id == "doc-123"
        assert chunk.content == "这是分块内容"
        assert chunk.start_index == 0
        assert chunk.end_index == 100
        assert chunk.chunk_index == 1
        assert chunk.type == ChunkType.TEXT
    
    def test_chunk_record_creation(self):
        """测试分块记录创建"""
        chunk = Chunk(
            id="chunk-456",
            document_id="doc-123",
            content="分块内容",
            start_index=0,
            end_index=50,
            chunk_index=1
        )
        
        chunk_record = ChunkRecord(
            id="record-789",
            chunk=chunk,
            dense_embedding=[0.1, 0.2, 0.3],
            sparse_embedding={"word1": 0.5, "word2": 0.3}
        )
        
        assert chunk_record.id == "record-789"
        assert chunk_record.chunk.id == "chunk-456"
        assert chunk_record.dense_embedding == [0.1, 0.2, 0.3]
        assert chunk_record.sparse_embedding == {"word1": 0.5, "word2": 0.3}
    
    def test_search_request_response(self):
        """测试搜索请求和响应"""
        # 创建搜索请求
        request = SearchRequest(
            query="测试查询",
            top_k=5,
            use_dense=True,
            use_sparse=True
        )
        
        assert request.query == "测试查询"
        assert request.top_k == 5
        assert request.use_dense is True
        assert request.use_sparse is True
        
        # 创建模拟的分块记录
        chunk = Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="相关内容",
            start_index=0,
            end_index=50,
            chunk_index=1
        )
        
        chunk_record = ChunkRecord(
            id="record-1",
            chunk=chunk
        )
        
        # 创建查询结果
        result = QueryResult(
            chunk_record=chunk_record,
            score=0.95,
            retrieval_method="hybrid"
        )
        
        # 创建搜索响应
        response = SearchResponse(
            results=[result],
            total_results=1,
            search_time_ms=150.5
        )
        
        assert len(response.results) == 1
        assert response.results[0].score == 0.95
        assert response.total_results == 1
        assert response.search_time_ms == 150.5
    
    def test_ingestion_progress(self):
        """测试数据摄取进度"""
        progress = IngestionProgress(
            document_id="doc-123",
            stage="processing",
            progress=0.5,
            status="in_progress",
            message="正在处理文档"
        )
        
        assert progress.document_id == "doc-123"
        assert progress.stage == "processing"
        assert progress.progress == 0.5
        assert progress.status == "in_progress"
        assert progress.message == "正在处理文档"
    
    def test_trace_context(self):
        """测试追踪上下文"""
        trace = TraceContext(
            trace_id="trace-123",
            operation="search",
            stages=[
                {"stage": "query_processing", "duration_ms": 10},
                {"stage": "retrieval", "duration_ms": 50}
            ]
        )
        
        assert trace.trace_id == "trace-123"
        assert trace.operation == "search"
        assert len(trace.stages) == 2
        assert trace.stages[0]["stage"] == "query_processing"
    
    def test_error_response(self):
        """测试错误响应"""
        error = ErrorResponse(
            error="配置验证失败",
            code="CONFIG_VALIDATION_ERROR",
            details={"field": "api_key", "issue": "missing"}
        )
        
        assert error.error == "配置验证失败"
        assert error.code == "CONFIG_VALIDATION_ERROR"
        assert error.details["field"] == "api_key"
    
    def test_document_metadata(self):
        """测试文档元数据"""
        doc = Document(
            id="doc-456",
            title="带元数据的文档",
            content="内容",
            type=DocumentType.PDF,
            metadata={
                "author": "测试作者",
                "pages": 10,
                "keywords": ["测试", "文档"]
            }
        )
        
        assert doc.metadata["author"] == "测试作者"
        assert doc.metadata["pages"] == 10
        assert doc.metadata["keywords"] == ["测试", "文档"]


if __name__ == "__main__":
    pytest.main([__file__])