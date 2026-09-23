"""HTTP 路由层：对外 API 与前端静态资源。"""
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from sse_starlette import EventSourceResponse

import app.llm as llm  # noqa: F401  (确保模块可打桩的同时被显式依赖)
from app import __version__
from app.config import settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocListResponse,
    DocumentItem,
    HealthResponse,
    UploadResponse,
)
from app.state import state

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


# ==================================================================
# 业务接口
# ==================================================================
@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档：解析 → 分块 → 向量化 → 持久化 → 重建索引与 Agent。"""
    name = (file.filename or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型 {ext}，仅支持 {sorted(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")

    try:
        added = state.processor.add_document(name, content)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("文档处理失败")
        raise HTTPException(status_code=502, detail=f"文档处理失败：{exc}") from exc

    try:
        state.rebuild_after_change()
    except Exception as exc:  # noqa: BLE001
        logger.exception("索引/Agent 重建失败")
        raise HTTPException(status_code=502, detail=f"索引构建失败：{exc}") from exc

    return UploadResponse(
        filename=name,
        status="ok",
        chunks=added,
        total_chunks=len(state.processor.chunks),
    )


@router.get("/documents", response_model=DocListResponse)
async def list_documents():
    """文档清单。"""
    return DocListResponse(
        documents=[DocumentItem(**d) for d in state.processor.documents]
    )


@router.delete("/documents/{filename}", response_model=DeleteResponse)
async def delete_document(filename: str):
    """删除文档（级联清理分块向量并重建索引）。"""
    ok = state.processor.delete_document(filename)
    if not ok:
        raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在")
    if state.processor.chunks:
        state.rebuild_after_change()
    else:
        state.rag.reset()
    return DeleteResponse(filename=filename, deleted=True)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """同步问答（Agent 完整链路）。"""
    if state.agent is None:
        raise HTTPException(status_code=400, detail="请先上传文档再提问")
    config = {"configurable": {"thread_id": req.conversation_id}}
    try:
        result = state.agent.invoke({"messages": [("user", req.question)]}, config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("问答调用失败")
        raise HTTPException(status_code=502, detail=f"问答服务异常：{exc}") from exc
    answer = result["messages"][-1].content
    return ChatResponse(answer=answer, request_id=str(uuid.uuid4()), sources=[])


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式问答（SSE 逐字输出）。"""
    if state.agent is None:
        raise HTTPException(status_code=400, detail="请先上传文档再提问")
    config = {"configurable": {"thread_id": req.conversation_id}}

    async def generate():
        try:
            async for event in state.agent.astream_events(
                {"messages": [("user", req.question)]},
                config=config,
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    token = event["data"]["chunk"].content
                    if token:
                        yield {"data": token}
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式问答异常")
            yield {"data": f"\n[错误] {exc}"}
        yield {"data": "[DONE]"}

    return EventSourceResponse(generate())


# ==================================================================
# 系统接口
# ==================================================================
@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查：暴露运行状态与脱敏配置，便于监控与排障。"""
    return HealthResponse(
        status="ok",
        version=__version__,
        model=settings.CHAT_MODEL,
        documents=len(state.processor.documents),
        chunks=len(state.processor.chunks),
        config=settings.as_dict(),
        indexes={
            "faiss": state.rag.index_ready,
            "bm25": state.rag.index_ready,
            "agent": state.agent is not None,
        },
    )


__all__ = ["router"]