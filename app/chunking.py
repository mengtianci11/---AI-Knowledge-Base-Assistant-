"""文本分块策略（RAG 检索质量的第一道关卡）。

- fixed: 固定滑窗切分（原版逻辑），实现简单、参数可复现；
- recursive: 段落感知的递归分块（推荐），先按空行聚合段落，
  再按 chunk_size 合并，超长段回落滑窗切分 —— 尽量保住语义完整，
  减少「一句话被拦腰截断」导致的检索噪声。
"""
import re
from typing import List

SEPARATORS = ["\n\n", "\n", "。"]


def split_text_fixed(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """固定窗口滑动切分。"""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("参数不合法：需满足 chunk_size > 0 且 0 <= overlap < chunk_size")
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def split_text_recursive(
    text: str, chunk_size: int = 300, overlap: int = 50, seps: List[str] | None = None
) -> List[str]:
    """段落感知递归分块：先按段落聚合，超长再滑窗。"""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("参数不合法：需满足 chunk_size > 0 且 0 <= overlap < chunk_size")
    seps = seps or SEPARATORS
    text = text.strip()
    if not text:
        return []

    # 1) 按段落切分（保留边界信息）
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # 2) 逐步合并段落，超过容量则落块
    chunks: List[str] = []
    buffer = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            # 超长段落：先落缓冲，再滑窗切该段
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(split_text_fixed(para, chunk_size, overlap))
            continue
        if not buffer:
            buffer = para
        elif len(buffer) + 1 + len(para) <= chunk_size:
            buffer = f"{buffer}\n{para}"
        else:
            chunks.append(buffer)
            buffer = para
    if buffer:
        chunks.append(buffer)

    # 3) 可选：对块做轻量后处理，缝合过短的尾巴
    merged: List[str] = []
    for c in chunks:
        if merged and len(merged[-1]) + 1 + len(c) <= chunk_size:
            merged[-1] = f"{merged[-1]}\n{c}"
        else:
            merged.append(c)
    return [c.strip() for c in merged if c.strip()]


def split_text(  # noqa: D103
    text: str, strategy: str = "recursive", chunk_size: int = 300, overlap: int = 50
) -> List[str]:
    if strategy == "fixed":
        return split_text_fixed(text, chunk_size, overlap)
    return split_text_recursive(text, chunk_size, overlap)


__all__ = ["split_text", "split_text_fixed", "split_text_recursive"]