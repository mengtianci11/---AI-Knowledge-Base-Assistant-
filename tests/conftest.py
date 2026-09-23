"""pytest 公共夹具：离线化（不打外部 API）+ 隔离数据目录。

所有测试通过 monkeypatch 替换 embedding / rerank / agent，
保证在没有网络和 API Key 的环境下也能完整跑通。
"""
import hashlib
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fake_embed(text: str) -> list[float]:
    """确定性伪向量（8 维），用于离线测试。"""
    h = hashlib.md5(text.encode("utf-8")).digest()
    return [((b % 20) - 10) / 10.0 for b in h[:8]]


class FakeAgent:
    """不联网的假 Agent，固定输出 + 流式逐字。"""

    def invoke(self, messages, config):
        return {"messages": [types.SimpleNamespace(content="这是一条离线测试回答。")]}

    async def astream_events(self, messages, config=None, version="v2"):
        for ch in "测试流式":
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": types.SimpleNamespace(content=ch)},
            }
        yield {"event": "on_chain_end", "data": {}}


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch):
    """全局：注入假 Key、关闭外部 Rerank、打桩向量化，保证测试离线。"""
    from app import config
    import app.llm as llm

    monkeypatch.setattr(config.settings, "API_KEY", "sk-test-fake")
    monkeypatch.setattr(config.settings, "RERANK_ENABLED", False)
    monkeypatch.setattr(config.settings, "RAG_FUSION", False)
    monkeypatch.setattr(llm, "embed_text", fake_embed)
    monkeypatch.setattr(llm, "embed_texts", lambda texts: [fake_embed(t) for t in texts])
    return config.settings


@pytest.fixture
def doc_store(tmp_path):
    from app.db import DocumentStore

    return DocumentStore(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def processor(doc_store, monkeypatch):
    """隔离的文档处理器（伪向量化）。"""
    from app.document_processor import DocumentProcessor

    p = DocumentProcessor(doc_store)
    monkeypatch.setattr(p, "embed_fn", fake_embed)
    return p


@pytest.fixture
def rag(processor):
    from app.retrieval import AdvancedRAG

    return AdvancedRAG(processor)


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """隔离的 FastAPI 测试客户端（tmp 数据库 + 假 Agent + 伪向量）。"""
    import app.routes as routes
    from app.agent import build_agent  # noqa: F401
    from app.db import DocumentStore
    from app.document_processor import DocumentProcessor
    from app.main import create_app
    from app.retrieval import AdvancedRAG

    store = DocumentStore(db_path=str(tmp_path / "api.db"))
    processor = DocumentProcessor(store)
    monkeypatch.setattr(processor, "embed_fn", fake_embed)
    rag_inst = AdvancedRAG(processor)

    from app import state as state_mod

    for attr, val in [("store", store), ("processor", processor), ("rag", rag_inst), ("agent", None)]:
        setattr(state_mod.state, attr, val)
    monkeypatch.setattr(routes, "state", state_mod.state)

    app = create_app(with_persistence=False)
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    client = TestClient(app)
    client.state_mod = state_mod.state  # type: ignore[attr-defined]
    return client