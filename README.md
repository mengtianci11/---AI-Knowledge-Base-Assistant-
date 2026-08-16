# ---AI-Knowledge-Base-Assistant-
基于企业知识库API（如RAG技术）开发智能问答模块，实现用户问题的精准检索与回答。可用于企业内部文档散落各处（PDF、Word、Markdown），员工找资料靠问人、翻共享文件夹时的AI助手，上传文档后直接提问，精准回答并带来源引用，查不到时转人工。也可用于Agent开发中的学习实例
```
# 企业知识库 AI 助手

基于 RAG + Agent 构建的企业级智能问答系统，支持 PDF/Markdown 文档上传、混合检索、流式对话与工具调度，开箱即用的知识库解决方案。

##  项目简介

企业内部文档往往散落在各处，员工查找资料依赖人工询问和文件夹翻阅，效率低下且信息不一致。本项目打造了一个**企业知识库 AI 助手**：上传文档后即可直接提问，系统精准回答并带来源引用，知识库无法解答时自动转接人工，大幅提升企业内部信息获取效率。

##  系统架构


```
<img width="2880" height="1344" alt="image" src="https://github.com/user-attachments/assets/45b32dfd-9d7a-48fd-9193-2e6a573128ed" />
```


##  核心功能

- **多格式文档上传**：支持 `.txt`、`.md`、`.pdf` 等格式文档一键上传解析
- **多级 RAG 检索管线**：查询改写 → 向量+BM25 混合检索 → RRF 融合 → Rerank 精排
- **智能 Agent 调度**：基于 LangGraph 自动识别意图，选择知识库检索/数学计算/转人工
- **SSE 流式输出**：逐 token 推送，打字机效果，流畅的对话体验
- **来源引用追溯**：回答自动标注引用来源 `[1][2]`，可追溯原文
- **相关性过滤**：低于阈值自动判定"未找到答案"，有效减少幻觉
- **多轮对话记忆**：支持上下文指代消解，对话连贯自然
- **全链路日志**：请求 ID 贯穿全程，便于问题排查与效果优化
- **Docker 一键部署**：开箱即用，快速部署到生产环境

##  技术栈

| 模块 | 技术选型 |
|------|----------|
| 后端框架 | FastAPI |
| Agent 编排 | LangGraph / LangChain |
| 向量检索 | FAISS |
| 关键词检索 | BM25 |
| 重排模型 | SiliconFlow Rerank API |
| 文档解析 | pypdf / pdfplumber |
| 流式传输 | SSE (Server-Sent Events) |
| 容器化 | Docker + docker-compose |
| 前端 | 原生 HTML + JavaScript |

##  技术亮点

### 1. 混合检索策略
纯向量检索对专有名词、编号类查询敏感度不足，引入 BM25 关键词检索补充精确匹配能力，通过 RRF（倒数排名融合）算法合并两路结果，兼顾语义匹配与字面匹配。

### 2. Rerank 精排提升准确率
粗召回 Top-20 结果中真正相关的文档可能排名靠后，通过 Cross-Encoder 重排模型进行精细化相关性打分，显著提升 Top-K 准确率。

### 3. 查询改写优化召回
用户口语化提问与文档书面语存在用词差异，通过 LLM 对原始问题进行改写、扩展和澄清，缩小语义鸿沟，提升召回率。

### 4. Agent 自主决策
基于 Function Calling 机制，模型自主判断问题类型：知识库问题走检索、数学问题调计算器、超范围问题转人工，无需人工规则硬编码。

### 5. 可靠性保障
- 相关性阈值过滤（0.3 分以下丢弃），从源头抑制幻觉
- 统一超时控制与异常捕获，API 故障时友好降级
- 全链路请求 ID 追踪，完整记录用户问题、命中片段、工具调用与耗时

##  项目结构

```

kb_assistant/
├── app/
│   ├── **init**.py
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 全局配置
│   ├── api/
│   │   ├── chat.py             # 聊天接口（同步 + 流式）
│   │   └── documents.py        # 文档上传与管理接口
│   ├── core/
│   │   ├── rag.py              # 进阶 RAG 检索管线
│   │   ├── agent.py            # LangGraph Agent
│   │   └── tools.py            # 工具定义
│   ├── models/
│   │   └── schemas.py          # Pydantic 数据模型
│   └── services/
│       └── document_processor.py  # 文档解析与分块
├── data/
│   └── documents/              # 上传文档存储目录
├── static/                     # 前端页面
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md

```

##  快速开始

### 方式一：本地运行

1. **克隆仓库**
```bash
git clone https://github.com/your-username/kb-assistant.git
cd kb-assistant
```

2. **安装依赖**

```
pip install -r requirements.txt
```

3. **配置环境变量**

```
cp .env.example .env
# 编辑 .env，填入 SiliconFlow API Key 等配置
```

4. **启动服务**

```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5. **访问页面**

打开浏览器访问 `http://localhost:8000` 即可使用。

### 方式二：Docker 一键启动

```
docker-compose up -d
```

服务启动后访问 `http://localhost:8000`。

##  接口说明

### 文档上传

```
POST /documents/upload
Content-Type: multipart/form-data
```

上传文档并自动解析分块、构建索引。

### 获取文档列表

```
GET /documents
```

返回所有已上传的文档列表。

### 同步问答

```
POST /chat
Content-Type: application/json

{
  "query": "怎么申请年假？",
  "thread_id": "conv-001"
}
```

### 流式问答

```
POST /chat/stream
Content-Type: application/json

{
  "query": "怎么申请年假？",
  "thread_id": "conv-001"
}
```

通过 SSE 逐 token 返回回答内容与工具调用状态。

##  验收标准

- 上传 2-3 份文档后，可准确回答文档内问题，回答附带来源引用
- 多轮对话中 "它"" 这个 " 等指代可正确消解
- 知识库无答案时明确告知，不编造内容
- 数学计算类问题自动调用计算器工具
- 前端流式打字机效果流畅
- `docker-compose up` 单命令即可完整启动
- 全链路日志可追溯

##  简历参考

> 
> **企业知识库 AI 助手** | 个人项目
> 
> 
> - 基于 FastAPI + LangGraph 构建企业知识库问答系统，支持 PDF/Markdown 文档上传、混合检索、流式对话
> - 实现查询改写 + 向量 / BM25 混合检索 + Rerank 重排的多级 RAG 管线，检索准确率较基础 RAG 显著提升
> - 设计 Agent 工具调度机制，支持知识库检索、数学计算、人工转接，自动识别用户意图
> - 实现 SSE 流式输出、全链路日志追踪、相关性阈值过滤（防幻觉）、Docker 一键部署
> - 技术栈：Python, LangChain/LangGraph, FAISS, FastAPI, Docker, SiliconFlow API

##  License

MIT License
