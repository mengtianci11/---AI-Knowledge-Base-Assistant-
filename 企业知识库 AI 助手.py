"""
企业知识库 AI 助手

安装：pip install fastapi uvicorn pydantic python-dotenv sse-starlette
      pip install openai faiss-cpu rank-bm25 langchain langchain-openai langgraph
      pip install pypdf python-multipart jieba
"""
import os
import io
import json
import time
import uuid
import logging
import requests
import numpy as np
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from sse_starlette import EventSourceResponse

load_dotenv()

# ============================================================
# 配置
# ============================================================
API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-nvptsousxmttzmqrompdpbdjjhojkfkiwvtdikqmtgdhzgdo")
BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-ai/DeepSeek-V3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DATA_DIR = os.getenv("DATA_DIR", "data/documents")
os.makedirs(DATA_DIR, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("app.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================
#  1: 文档处理
# ============================================================

class DocumentProcessor:
    """文档解析、分块、向量化"""

    def __init__(self):
        self.chunks = []  # [{"text", "embedding", "source", "chunk_id"}]
        self.documents = []  # 文档清单

    def parse_file(self, filename: str, content: bytes) -> str:
        """解析文件，提取纯文本。支持 .txt/.md/.pdf"""
        if filename.endswith('.pdf'):
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() for page in reader.pages)
        else:  # txt/md
            return content.decode('utf-8')

    def split_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> list:
        """文本分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def add_document(self, filename: str, content: bytes):
        """完整流程：解析 → 分块 → 向量化 → 存储"""
        text = self.parse_file(filename, content)
        pieces = self.split_text(text)
        for i, piece in enumerate(pieces):
            emb = self.get_embedding(piece)
            self.chunks.append({
                "text": piece, "embedding": emb,
                "source": filename, "chunk_id": f"{filename}_{i}"
            })
        self.documents.append({"name": filename, "chunks": len(pieces), "added_at": datetime.now().isoformat()})
        logger.info(f"文档 {filename} 已添加，共 {len(pieces)} 块")

    @staticmethod
    def get_embedding(text: str) -> list:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding


# ============================================================
#  2: RAG 检索管线
# ============================================================

class AdvancedRAG:
    """进阶RAG：查询改写 + 混合检索 + Rerank"""

    def __init__(self, doc_processor: DocumentProcessor):
        self.dp = doc_processor
        self.vectorstore = None  # FAISS 向量库，文档上传后构建
        self.bm25 = None         # BM25 索引，文档上传后构建
        self.docs = []           # BM25 对应的文档文本列表

    def build_indices(self):
        """从 doc_processor.chunks 构建 FAISS 向量库和 BM25 索引"""
        if not self.dp.chunks:
            logger.warning("没有文档数据，无法构建索引")
            return

        # ---- 构建 FAISS 向量索引 ----
        from langchain_core.documents import Document
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL
        )

        langchain_docs = []
        for chunk in self.dp.chunks:
            langchain_docs.append(Document(
                page_content=chunk["text"],
                metadata={"source": chunk["source"], "chunk_id": chunk["chunk_id"]}
            ))

        # 直接用已有的 embedding 向量构建 FAISS
        texts = [d.page_content for d in langchain_docs]
        metadatas = [d.metadata for d in langchain_docs]
        embeddings_array = np.array([chunk["embedding"] for chunk in self.dp.chunks])

        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings_array.tolist())),
            embedding=embeddings,
            metadatas=metadatas
        )
        logger.info(f"FAISS 向量库已构建，共 {len(texts)} 个文档块")

        # ---- 构建 BM25 索引 ----
        import jieba
        from rank_bm25 import BM25Okapi

        self.docs = [chunk["text"] for chunk in self.dp.chunks]
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 索引已构建，共 {len(self.docs)} 个文档块")

    def query_rewrite(self, question: str) -> list:
        """查询改写"""
        prompt = f"""你是一个搜索专家。把下面这个问题改写为2-3个不同角度的搜索查询。
每个查询使用不同的关键词组合，从不同侧面覆盖问题。
直接输出改写后的查询，每行一个，不要编号，不要其他内容。
问题：{question}"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        text = response.choices[0].message.content
        queries = [q.strip() for q in text.split('\n') if q.strip()]
        return queries

    def vector_search(self, query: str, top_k: int = 10) -> list:
        """向量检索"""
        if self.vectorstore is None:
            return []

        docs_with_score = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k
        )

        result = []
        for doc, similarity_score in docs_with_score:
            result.append({
                "content": doc.page_content,
                "score": float(similarity_score),
                "metadata": doc.metadata
            })

        return result

    def bm25_search(self, query: str, top_k: int = 10) -> list:
        """BM25检索"""
        if self.bm25 is None:
            return []

        import jieba
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = scores.argsort()[::-1][:top_k]

        result = []
        for idx in top_k_indices:
            result.append({
                "content": self.docs[idx],
                "score": float(scores[idx]),
                "metadata": {"doc_index": int(idx)}
            })

        return result

    def rrf_merge(self, results_list: list, k: int = 60, top_k: int = 20) -> list:
        """RRF合并"""
        doc_score_map = {}
        doc_info_map = {}

        for one_result_list in results_list:
            for rank, doc_dict in enumerate(one_result_list):
                current_rank = rank + 1
                rrf_score = 1 / (k + current_rank)
                doc_key = doc_dict["content"]

                if doc_key in doc_score_map:
                    doc_score_map[doc_key] += rrf_score
                else:
                    doc_score_map[doc_key] = rrf_score
                    doc_info_map[doc_key] = doc_dict

        # 注意：return 必须在 for 循环外面
        sorted_docs = sorted(
            doc_score_map.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_result = []
        for doc_key, total_score in sorted_docs[:top_k]:
            doc_info = doc_info_map[doc_key].copy()
            doc_info["score"] = total_score
            final_result.append(doc_info)

        return final_result

    def rerank(self, query: str, documents: list, top_n: int = 5) -> list:
        """硅基流动Rerank API"""
        doc_contents = [doc["content"] for doc in documents]

        if not doc_contents:
            return []

        try:
            url = f"{BASE_URL}/rerank"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": RERANK_MODEL,
                "query": query,
                "documents": doc_contents,
                "return_documents": False
            }

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            ranked_results = result["results"]
            top_results = ranked_results[:top_n]

            final_docs = []
            for item in top_results:
                original_index = item["index"]
                ranked_doc = documents[original_index].copy()
                ranked_doc["relevance_score"] = item["relevance_score"]
                final_docs.append(ranked_doc)

            return final_docs

        except Exception as e:
            logger.warning(f"重排序调用失败，降级使用原文档前 {top_n} 条：{str(e)}")
            return documents[:top_n]

    def retrieve(self, question: str) -> list:
        """完整检索流水线，返回Top-5带rerank_score的结果"""
        # 检查是否有文档索引
        if self.vectorstore is None or self.bm25 is None:
            return []

        rewrite_queries = self.query_rewrite(question)
        if question not in rewrite_queries:
            rewrite_queries.insert(0, question)

        all_retrieval_results = []
        for query in rewrite_queries:
            vector_docs = self.vector_search(query, top_k=10)
            all_retrieval_results.append(vector_docs)

            bm25_docs = self.bm25_search(query, top_k=10)
            all_retrieval_results.append(bm25_docs)

        merged_docs = self.rrf_merge(all_retrieval_results, top_k=20)
        final_docs = self.rerank(question, merged_docs, top_n=5)

        for doc in final_docs:
            if "relevance_score" in doc:
                doc["rerank_score"] = doc.pop("relevance_score")

        return final_docs


# ============================================================
#  3: Agent + 工具
# ============================================================

def build_agent(rag: AdvancedRAG):
    """构建LangGraph Agent"""
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver

    llm = ChatOpenAI(model=CHAT_MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0.3)

    @tool
    def search_knowledge_base(query: str) -> str:
        """检索企业知识库，查询公司制度、产品说明、技术文档等内部资料。当用户问到公司相关问题时使用。"""
        try:
            docs = rag.retrieve(query)

            if not docs:
                return "知识库中未检索到与问题相关的内容。"

            result_text = "知识库检索到的参考内容如下：\n"
            for idx, doc in enumerate(docs, start=1):
                source = doc.get("metadata", {}).get("source", "未知来源")
                result_text += f"\n【参考资料 {idx}】(来源: {source})\n{doc['content']}\n"

            return result_text

        except Exception as e:
            return f"知识库检索暂时不可用，错误原因：{str(e)}"

    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式，如 '3000 * 0.8'。"""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except ZeroDivisionError:
            return "错误：除数不能为零"
        except SyntaxError:
            return "错误：表达式语法有误，请检查运算符号与括号格式"
        except Exception as e:
            return f"计算失败：{str(e)}"

    @tool
    def transfer_to_agent(summary: str) -> str:
        """当问题无法通过知识库回答，或用户明确要求人工服务时，转接人工客服。输入问题摘要。"""
        return f"已为您转接人工客服。问题摘要：{summary}"

    tools = [search_knowledge_base, calculator, transfer_to_agent]

    system_prompt = """你是企业知识库AI助手。
- 公司制度、产品、技术问题 → 用 search_knowledge_base 检索后回答，引用来源
- 数学计算 → 用 calculator
- 知识库查不到或用户要投诉/找人工 → 用 transfer_to_agent
- 简单问候直接回答
回答要简洁专业，引用资料时标注来源。"""

    agent = create_react_agent(
        llm, tools,
        prompt=system_prompt,
        checkpointer=MemorySaver()
    )
    return agent


# ============================================================
#  4: FastAPI 接口
# ============================================================

app = FastAPI(title="企业知识库AI助手", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 托管前端静态文件
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
async def index():
    return FileResponse("index.html")


# 全局实例
doc_processor = DocumentProcessor()
rag = AdvancedRAG(doc_processor)
agent = None  # 在文档上传后初始化


class ChatRequest(BaseModel):
    question: str
    conversation_id: str = "default"


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """文档上传接口"""
    content = await file.read()
    doc_processor.add_document(file.filename, content)

    # 上传后重建检索索引
    rag.build_indices()

    # 初始化/重建 Agent
    global agent
    agent = build_agent(rag)

    return {
        "filename": file.filename,
        "status": "ok",
        "chunks": len(doc_processor.chunks)
    }


@app.get("/documents")
async def list_documents():
    """文档列表"""
    return {"documents": doc_processor.documents}


@app.post("/chat")
async def chat(req: ChatRequest):
    """同步问答"""
    if agent is None:
        raise HTTPException(status_code=400, detail="请先上传文档文档")

    config = {"configurable": {"thread_id": req.conversation_id}}
    result = agent.invoke({"messages": [("user", req.question)]}, config)
    answer = result["messages"][-1].content

    return {
        "answer": answer,
        "request_id": str(uuid.uuid4()),
        "sources": []
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式问答 SSE"""
    if agent is None:
        raise HTTPException(status_code=400, detail="请先上传文档")

    config = {"configurable": {"thread_id": req.conversation_id}}

    async def generate():
        async for event in agent.astream_events(
            {"messages": [("user", req.question)]},
            config=config,
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield {"data": token}
        yield {"data": "[DONE]"}

    return EventSourceResponse(generate())


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": CHAT_MODEL,
        "documents": len(doc_processor.documents),
        "chunks": len(doc_processor.chunks)
    }


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
