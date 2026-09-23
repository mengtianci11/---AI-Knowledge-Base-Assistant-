"""应用级共享状态（单例注册表）。

用 app.state 注入路由更符合 FastAPI 惯例，但为了兼容
原项目的全局访问方式并方便测试替换，这里提供轻量模块级注册表。
"""
from app.agent import build_agent
from app.db import DocumentStore
from app.document_processor import DocumentProcessor
from app.retrieval import AdvancedRAG


class AppState:
    """服务运行时组合根（composition root）。"""

    def __init__(self) -> None:
        self.store: DocumentStore = DocumentStore()
        self.processor: DocumentProcessor = DocumentProcessor(self.store)
        self.rag: AdvancedRAG = AdvancedRAG(self.processor)
        self.agent = None  # 上传文档后惰性构建

    # ------------------------------------------------------------------
    def restore(self) -> None:
        """启动时从持久化存储恢复（重启不丢文档）。"""
        self.processor.rebuild_from_store()
        if self.processor.chunks:
            self.rag.build_indices()
            self.agent = build_agent(self.rag)
            print(f"[恢复] 已加载 {len(self.processor.documents)} 份文档 / {len(self.processor.chunks)} 个分块")

    def rebuild_after_change(self) -> None:
        self.rag.build_indices()
        self.agent = build_agent(self.rag)

    def reset(self) -> None:
        self.processor.clear_all()
        self.rag.reset()
        self.agent = None


state = AppState()


__all__ = ["state"]