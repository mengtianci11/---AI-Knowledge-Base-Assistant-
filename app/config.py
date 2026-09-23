"""全局配置中心：从 .env 加载、集中校验，杜绝硬编码密钥。

所有可调参数（模型、地址、分块策略、服务端口…）都从这里读取，
保证「配置与代码分离」，这也是企业级工程的基本要求。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（app/ 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


def _get(name: str, default: str = "") -> str:
    """读取环境变量并去除首尾空白。"""
    return os.getenv(name, default).strip()


class Settings:
    """应用配置。"""

    def __init__(self) -> None:
        # ---- 模型服务（硅基流动 / OpenAI 兼容接口）----
        self.API_KEY = _get("SILICONFLOW_API_KEY")
        self.BASE_URL = _get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.CHAT_MODEL = _get("CHAT_MODEL", "deepseek-ai/DeepSeek-V3")
        self.EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.RERANK_MODEL = _get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

        # ---- 存储与持久化 ----
        self.DATA_DIR = BASE_DIR / _get("DATA_DIR", "data")
        self.DB_PATH = self.DATA_DIR / "kb.db"

        # ---- 服务 ----
        self.HOST = _get("HOST", "127.0.0.1")
        self.PORT = int(_get("PORT", "8000"))

        # ---- 分块与检索 ----
        self.CHUNK_STRATEGY = _get("CHUNK_STRATEGY", "recursive")  # fixed | recursive
        self.CHUNK_SIZE = int(_get("CHUNK_SIZE", "300"))
        self.CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "50"))
        self.TOP_K = int(_get("TOP_K", "5"))

        # ---- 检索增强 ----
        self.RAG_FUSION = _get("RAG_FUSION", "true").lower() == "true"  # 查询改写开关
        self.RERANK_ENABLED = _get("RERANK_ENABLED", "true").lower() == "true"

        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """启动前校验，缺配置时给出可操作的错误信息（fail-fast）。"""
        if not self.API_KEY:
            raise RuntimeError(
                "未检测到 SILICONFLOW_API_KEY。\n"
                f"请在项目根目录的 {ENV_FILE.name} 文件中添加：\n"
                "    SILICONFLOW_API_KEY=sk-你的密钥\n"
                "密钥可到 https://cloud.siliconflow.cn 获取。"
            )
        if not self.API_KEY.startswith("sk-"):
            raise RuntimeError(
                f"SILICONFLOW_API_KEY 格式异常（应以 sk- 开头）。当前值已隐去，请检查 {ENV_FILE.name}。"
            )
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise RuntimeError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE，请检查 .env 配置。")

    def as_dict(self) -> dict:
        """脱敏后的配置快照（可安全打印/写入健康检查）。"""
        return {
            "base_url": self.BASE_URL,
            "chat_model": self.CHAT_MODEL,
            "embedding_model": self.EMBEDDING_MODEL,
            "rerank_model": self.RERANK_MODEL,
            "chunk_strategy": self.CHUNK_STRATEGY,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "top_k": self.TOP_K,
            "rag_fusion": self.RAG_FUSION,
            "rerank_enabled": self.RERANK_ENABLED,
            "api_key_configured": bool(self.API_KEY),
        }


settings = Settings()