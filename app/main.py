"""FastAPI 应用组装：中间件、静态资源、路由、启动恢复。

升级点：
- 启动时自动从 SQLite 恢复文档与索引（重启不丢数据）；
- 配置集中校验（缺 Key / 参数非法时 fail-fast 并给出修复指引）；
- 版本化健康检查，便于部署监控。
"""
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import BASE_DIR, settings
from app.routes import router
from app.state import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("app.main")


def create_app(with_persistence: bool = True) -> FastAPI:
    """创建应用实例；with_persistence=False 可用于纯测试场景。"""
    settings.validate()

    app = FastAPI(
        title="企业知识库 AI 助手",
        description=(
            "Agentic RAG 企业知识库问答系统：多路召回（向量+BM25）+ RRF 融合 + "
            "Cross-Encoder 重排序 + LangGraph ReAct Agent + SQLite 持久化 + SSE 流式输出。"
        ),
        version=__version__,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # 前端静态资源（index.html 位于项目根目录）
    app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(BASE_DIR / "index.html")

    if with_persistence:
        try:
            state.restore()
        except Exception as exc:  # noqa: BLE001
            logger.warning("启动恢复持久化数据失败（忽略，以空库继续）：%s", exc)

    logger.info("应用已启动：v%s | 模型 %s | 数据目录 %s", __version__, settings.CHAT_MODEL, settings.DATA_DIR)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)