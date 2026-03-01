"""核心数据类型/契约定义"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class DocumentType(str, Enum):
    """文档类型枚举"""
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ChunkType(str, Enum):
    """分块类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"


class Document(BaseModel):
    """文档数据模型"""
    
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="文档唯一标识")
    title: str = Field(..., description="文档标题")
    content: str = Field(..., description="文档内容")
    type: DocumentType = Field(..., description="文档类型")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    file_path: Optional[str] = Field(None, description="文件路径")
    file_size: Optional[int] = Field(None, description="文件大小（字节）")
    sha256_hash: Optional[str] = Field(None, description="文件SHA256哈希")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class Chunk(BaseModel):
    """文档分块数据模型"""
    
    model_config = ConfigDict(use_enum_values=True)
    
    id: str = Field(..., description="分块唯一标识")
    document_id: str = Field(..., description="所属文档ID")
    content: str = Field(..., description="分块内容")
    type: ChunkType = Field(default=ChunkType.TEXT, description="分块类型")
    
    # 位置信息
    start_index: int = Field(..., description="在文档中的起始位置")
    end_index: int = Field(..., description="在文档中的结束位置")
    chunk_index: int = Field(..., description="分块序号")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="分块元数据")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class ChunkRecord(BaseModel):
    """分块记录（用于存储和检索）"""
    
    id: str = Field(..., description="记录唯一标识")
    chunk: Chunk = Field(..., description="分块数据")
    
    # 向量表示
    dense_embedding: Optional[List[float]] = Field(None, description="稠密向量嵌入")
    sparse_embedding: Optional[Dict[str, float]] = Field(None, description="稀疏向量嵌入（TF-IDF）")
    
    # 存储信息
    vector_id: Optional[str] = Field(None, description="向量存储ID")
    bm25_id: Optional[str] = Field(None, description="BM25索引ID")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class QueryResult(BaseModel):
    """查询结果"""
    
    model_config = ConfigDict(use_enum_values=True)
    
    chunk_record: ChunkRecord = Field(..., description="匹配的分块记录")
    score: float = Field(..., description="匹配分数")
    retrieval_method: str = Field(..., description="检索方法")
    
    # 引用信息
    citation: Optional[str] = Field(None, description="引用信息")


class SearchRequest(BaseModel):
    """搜索请求"""
    
    model_config = ConfigDict(use_enum_values=True)
    
    query: str = Field(..., description="查询文本")
    collection_name: Optional[str] = Field(None, description="集合名称")
    
    # 检索参数
    top_k: int = Field(default=10, description="返回结果数量")
    use_dense: bool = Field(default=True, description="使用稠密检索")
    use_sparse: bool = Field(default=True, description="使用稀疏检索")
    use_rerank: bool = Field(default=False, description="使用重排序")
    
    # 过滤条件
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")


class SearchResponse(BaseModel):
    """搜索响应"""
    
    results: List[QueryResult] = Field(..., description="查询结果列表")
    total_results: int = Field(..., description="总结果数")
    search_time_ms: float = Field(..., description="搜索耗时（毫秒）")
    
    # 检索统计
    retrieval_stats: Dict[str, Any] = Field(default_factory=dict, description="检索统计信息")


class IngestionProgress(BaseModel):
    """数据摄取进度"""
    
    document_id: str = Field(..., description="文档ID")
    stage: str = Field(..., description="处理阶段")
    progress: float = Field(..., description="进度（0-1）")
    status: str = Field(..., description="状态")
    message: Optional[str] = Field(None, description="进度消息")
    
    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class TraceContext(BaseModel):
    """追踪上下文"""
    
    trace_id: str = Field(..., description="追踪ID")
    operation: str = Field(..., description="操作名称")
    
    # 阶段追踪
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="处理阶段")
    
    # 时间信息
    start_time: datetime = Field(default_factory=datetime.now, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="追踪元数据")


class ErrorResponse(BaseModel):
    """错误响应"""
    
    error: str = Field(..., description="错误信息")
    code: str = Field(..., description="错误代码")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    
    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.now, description="错误发生时间")