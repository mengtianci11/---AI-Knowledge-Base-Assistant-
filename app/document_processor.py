"""文档处理服务：解析 → 去重校验 → 分块 → 向量化 → 持久化。

升级点 vs 原版：
- 失败原子性：先分块向量化成功，再整体落库，避免「半入库」脏数据；
- MD5 内容去重 + 同名校验；
- 解析容错（PDF 无文本页、GBK 编码回退）；
- 支持删除文档（级联清理分块与向量）。
"""
import hashlib
import io
import logging
from typing import Callable

import pypdf

import app.llm as llm
from app.chunking import split_text
from app.config import settings
from app.db import DocumentStore

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """负责文档的解析、分块、向量化与持久化调度。"""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.chunks: list[dict] = []      # [{"text","embedding","source","chunk_id"}]
        self.documents: list[dict] = []   # 文档清单（与持久化一致）
        self.embed_fn: Callable[[str], list[float]] = staticmethod(llm.embed_text)

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------
    def parse_file(self, filename: str, content: bytes) -> str:
        """解析 .pdf / .txt / .md，统一返回纯文本。"""
        if filename.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(p for p in pages if p.strip())
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            logger.info("文件 %s 非 UTF-8 编码，按 GBK 回退解码", filename)
            return content.decode("gbk", errors="ignore")

    # ------------------------------------------------------------------
    # 向量化
    # ------------------------------------------------------------------
    def get_embedding(self, text: str) -> list[float]:
        return self.embed_fn(text)

    # ------------------------------------------------------------------
    # 分块
    # ------------------------------------------------------------------
    def split_text(self, text: str) -> list[str]:
        return split_text(text, settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    # ------------------------------------------------------------------
    # 增删改
    # ------------------------------------------------------------------
    def add_document(self, filename: str, content: bytes) -> int:
        """完整流程：解析 → 校验 → 分块 → 向量化 → 落库 → 刷新内存。返回新增块数。"""
        parsed = self.parse_file(filename, content)
        if not parsed.strip():
            raise ValueError(f"文档 {filename} 解析后内容为空，请检查文件格式")

        md5 = hashlib.md5(content).hexdigest()
        if self.store.doc_exists(md5, filename):
            raise ValueError(f"文档已存在（{filename} 或相同内容），请勿重复上传")

        pieces = self.split_text(parsed)
        if not pieces:
            raise ValueError(f"文档 {filename} 未切分出有效分块")

        # 先向量化全部成功，再统一落库（原子性）
        chunks: list[tuple[str, list[float]]] = []
        for piece in pieces:
            emb = self.get_embedding(piece)
            chunks.append((piece, emb))

        self.store.add_document(filename, md5, chunks)
        self.rebuild_from_store()
        logger.info("文档 %s 已入库：%d 个分块", filename, len(pieces))
        return len(pieces)

    def delete_document(self, filename: str) -> bool:
        ok = self.store.delete_document(filename)
        if ok:
            self.rebuild_from_store()
            logger.info("文档 %s 已删除", filename)
        return ok

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------
    def rebuild_from_store(self) -> None:
        """从持久化存储重建内存态（启动恢复 / 增删后刷新共用）。"""
        self.documents, self.chunks = self.store.load_all()

    def clear_all(self) -> None:
        self.store.clear_all()
        self.rebuild_from_store()


__all__ = ["DocumentProcessor"]