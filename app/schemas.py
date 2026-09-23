"""接口数据模型（Pydantic Schema）。"""
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    conversation_id: str = Field("default", max_length=64, description="会话 ID（用于多轮记忆）")


class ChatResponse(BaseModel):
    answer: str
    request_id: str
    sources: List[dict] = []


class UploadResponse(BaseModel):
    filename: str
    status: str = "ok"
    chunks: int = Field(..., description="本次上传新增的分块数")
    total_chunks: int = Field(..., description="库内全部分块数")


class DocumentItem(BaseModel):
    name: str
    chunks: int
    added_at: str


class DocListResponse(BaseModel):
    documents: List[DocumentItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    documents: int
    chunks: int
    config: dict
    indexes: dict


class DeleteResponse(BaseModel):
    filename: str
    deleted: bool


class ErrorResponse(BaseModel):
    detail: str