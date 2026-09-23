"""SQLite 持久化层：文档与分块落盘，服务重启后自动恢复（核心升级点之一）。

设计说明：
- 使用 stdlib sqlite3，零额外依赖；
- 分块向量以 BLOB 存储（pickle 序列化），重启后按需重建 FAISS/BM25 内存索引；
- WAL 模式 + 线程锁，兼顾并发读写安全；
- 文档按内容 MD5 去重，避免重复入库。
"""
import pickle
import sqlite3
import threading
from datetime import datetime
from typing import Optional

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL UNIQUE,
  md5         TEXT NOT NULL,
  chunk_count INTEGER NOT NULL,
  added_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks(
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  text        TEXT NOT NULL,
  embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class DocumentStore:
    """文档-分块-向量 的 SQLite 持久化仓库。"""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or str(settings.DB_PATH)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            conn.commit()

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------
    def doc_exists(self, md5: str, name: Optional[str] = None) -> bool:
        with self._connect() as conn:
            if name:
                row = conn.execute(
                    "SELECT 1 FROM documents WHERE md5 = ? OR name = ?", (md5, name)
                ).fetchone()
            else:
                row = conn.execute("SELECT 1 FROM documents WHERE md5 = ?", (md5,)).fetchone()
        return row is not None

    def add_document(self, name: str, md5: str, chunks: list[tuple[str, list[float]]]) -> int:
        """写入文档与其分块（text, embedding）。返回 doc_id。"""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO documents (name, md5, chunk_count, added_at) VALUES (?, ?, ?, ?)",
                (name, md5, len(chunks), datetime.now().isoformat()),
            )
            doc_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO chunks (doc_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
                [
                    (doc_id, i, text, sqlite3.Binary(pickle.dumps(emb)))
                    for i, (text, emb) in enumerate(chunks)
                ],
            )
            conn.commit()
        return doc_id

    def delete_document(self, name: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE name = ?", (name,))
            conn.commit()
        return cur.rowcount > 0

    def clear_all(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.commit()

    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------
    def load_all(self) -> tuple[list[dict], list[dict]]:
        """读取全部文档清单与分块（含 source 来源映射）。"""
        docs: list[dict] = []
        chunks: list[dict] = []
        with self._connect() as conn:
            for r in conn.execute("SELECT * FROM documents ORDER BY id"):
                docs.append(
                    {
                        "name": r["name"],
                        "chunks": r["chunk_count"],
                        "added_at": r["added_at"],
                        "md5": r["md5"],
                    }
                )
            for r in conn.execute(
                """
                SELECT c.*, d.name AS source
                FROM chunks c JOIN documents d ON d.id = c.doc_id
                ORDER BY c.id
                """
            ):
                chunks.append(
                    {
                        "text": r["text"],
                        "embedding": pickle.loads(r["embedding"]),
                        "source": r["source"],
                        "chunk_id": f"{r['doc_id']}_{r['chunk_index']}",
                    }
                )
        return docs, chunks

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": docs, "chunks": chunks}


__all__ = ["DocumentStore"]