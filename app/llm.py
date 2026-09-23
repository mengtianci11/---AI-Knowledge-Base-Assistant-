"""LLM 客户端封装：统一管理 embedding / chat / rerank 三类调用。

集中收口外部 API 依赖，方便：限流重试、日志追踪、单元测试打桩。
"""
import logging
import time

import requests
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# 惰性创建：首次使用时才实例化，便于测试替换
_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.API_KEY, base_url=settings.BASE_URL)
    return _client


def _retry(fn, retries: int = 3, base_delay: float = 1.0):
    """简单指数退避重试，应对瞬时网络抖动 / 限流（429）。"""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("外部模型调用失败（%s），%s 秒后重试（%d/%d）", exc, delay, attempt + 1, retries)
            time.sleep(delay)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化（支持空列表 / 去空白）。"""
    texts = [t for t in (x.strip() for x in texts) if t]
    if not texts:
        return []
    resp = _retry(lambda: get_client().embeddings.create(model=settings.EMBEDDING_MODEL, input=texts))
    return [d.embedding for d in resp.data]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def chat(messages: list[dict], temperature: float = 0.3, **kwargs) -> str:
    """Chat 补全，返回纯文本内容。"""
    resp = _retry(lambda: get_client().chat.completions.create(
        model=settings.CHAT_MODEL, messages=messages, temperature=temperature, **kwargs
    ))
    return resp.choices[0].message.content


def rerank(query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
    """重排序：按相关性对候选文档重新打分排序。

    调用失败时抛出异常，由上层（检索管线）降级为「原始顺序截断」。
    """
    doc_contents = [d["content"] for d in documents]
    if not doc_contents:
        return []

    url = f"{settings.BASE_URL}/rerank"
    headers = {"Authorization": f"Bearer {settings.API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": settings.RERANK_MODEL,
        "query": query,
        "documents": doc_contents,
        "return_documents": False,
    }
    resp = _retry(lambda: requests.post(url, headers=headers, json=payload, timeout=30))
    resp.raise_for_status()
    result = resp.json()

    final_docs = []
    for item in sorted(result["results"], key=lambda x: x.get("index"))[:top_n]:
        original_index = item["index"]
        ranked_doc = documents[original_index].copy()
        ranked_doc["relevance_score"] = item["relevance_score"]
        final_docs.append(ranked_doc)
    # 按相关性降序
    final_docs.sort(key=lambda d: d["relevance_score"], reverse=True)
    return final_docs