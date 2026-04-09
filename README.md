# PrismFlow 🔮

PrismFlow 是一个高性能、可观测的 **RAG（检索增强生成）网关**，采用 Go 语言编写。它旨在为 LLM 应用提供统一的接入层，内置语义缓存、混合检索、精排、流式响应以及全链路追踪能力。

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Go Version](https://img.shields.io/badge/Go-1.25+-00ADD8.svg)

## ✨ 核心特性

- **🚀 完整 RAG 编排**：Embedding → Hybrid Search → Rerank → Prompt Assembly → LLM Generation
- **🧠 语义缓存 (Semantic Cache)**：基于向量相似度缓存 LLM 响应（Redis Stack），降低延迟和 Token 成本
- **🔍 混合检索 (Hybrid Search)**：向量检索 + BM25 全文检索，使用 RRF (Reciprocal Rank Fusion) 融合
- **🎯 精排 (Rerank)**：BGE-Reranker-v2-m3 对召回结果二次排序，提升相关性
- **🌊 流式响应 (SSE)**：原生 Server-Sent Events，打字机式流畅体验
- **🔌 多模型支持**：
  - **LLM**: OpenAI, DeepSeek
  - **Embedding**: llama.cpp (nomic-embed-text) / Ollama / Mock
  - **VectorDB**: Milvus (HNSW + COSINE)
  - **Reranker**: BGE-Reranker-v2-m3 (TEI/Xinference) / Mock
- **👀 全链路可观测性**：OpenTelemetry + Jaeger，覆盖 Embedding、Search、Rerank、LLM 各阶段
- **🖥️ 内置 Web UI**：开箱即用的对话测试界面
- **📊 评估框架**：内置 Context Precision / Recall 评估，支持多策略对比实验

## 🏗️ 架构概览

```
User Query → SemanticCache → [miss] → Embedding → Hybrid Retrieval → Rerank → Prompt → LLM Stream → SSE
                  ↓ [hit]                          (Vector + BM25/RRF)  (BGE)
             Cached Response
```

详细架构设计见 [docs/architecture.md](docs/architecture.md)。

## 🛠️ 快速开始

### 前置要求

- Go 1.25+
- Docker & Docker Compose
- llama.cpp（本地 Embedding）或 Ollama

### 1. 启动基础设施

```bash
cd internal/deployment
docker-compose up -d
```

启动的服务：Redis Stack、Milvus、Jaeger

### 2. 启动 Embedding 服务

使用 llama.cpp（推荐）：

```bash
# 下载 nomic-embed-text 模型
curl -L -o ~/models/nomic-embed-text-v1.5.Q4_K_M.gguf \
  "https://hf-mirror.com/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"

# 启动 embedding 服务
llama-server --model ~/models/nomic-embed-text-v1.5.Q4_K_M.gguf --port 11434 --embedding
```

或使用 Ollama：

```bash
ollama pull nomic-embed-text
ollama serve
```

### 3. 配置

编辑 `configs/config.yaml`：

```yaml
llm:
  provider: "deepseek"
  api_key: "your-api-key"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com"

embedding:
  provider: "llamacpp"    # llamacpp / ollama / mock
  base_url: "http://localhost:11434"
  model: "nomic-embed-text"

rag:
  cache_threshold: 0.95   # 语义缓存相似度阈值
  score_threshold: 0.0    # 检索结果最低分数阈值
  top_k: 5

reranker:
  provider: "mock"        # bge / mock
  base_url: "http://localhost:8082"
  model: "bge-reranker-v2-m3"
  top_n: 3
```

### 4. 导入知识库

```bash
go run pkg/ingest/ingest.go -file your_document.md -chunk-size 500 -batch-size 5
```

支持的参数：
- `-file`：Markdown 文件路径
- `-chunk-size`：分块大小（字符数，默认 500）
- `-overlap`：分块重叠（字符数，默认 50）
- `-batch-size`：批量写入大小（默认 20）
- `-config`：配置文件路径（默认 `configs/config.yaml`）

Ingest 工具会自动按 Markdown 标题分段，在句子边界处分块，批量计算 embedding 并写入 Milvus。

### 5. 运行服务

```bash
go run cmd/main.go
```

- **Web UI**: http://localhost:8080
- **Chat API**: `POST http://localhost:8080/v1/chat`
- **Ingest API**: `POST http://localhost:8080/v1/ingest`
- **Health**: `GET http://localhost:8080/health`
- **Jaeger UI**: http://localhost:16686

## 📖 API

### Chat（流式 SSE）

```bash
curl -N http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是 redo log 和 binlog？"}'
```

响应格式：
```
data: {"content":"redo log"}
data: {"content":"是"}
...
data: {"done":true}
```

### Ingest

```bash
curl -X POST http://localhost:8080/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"content": "文档内容", "meta": {"title": "标题"}}]}'
```

## 📊 效果评估

内置评估框架，支持三种检索策略的对比实验：

```bash
go run eval/eval.go -config configs/config.yaml -testset eval/testset.json
```

| 实验 | 策略 |
|------|------|
| baseline | 纯向量检索 |
| +Rerank | 向量检索 + BGE Rerank |
| +Hybrid | Hybrid Search + BGE Rerank |

详见 [docs/evaluation.md](docs/evaluation.md)。

## 🔍 可观测性

访问 Jaeger UI (http://localhost:16686) 查看完整链路：

```
SemanticCache.Check → Embedding → Retrieval (Vector + BM25) → Rerank → PromptAssembly → LLMGeneration → SSE
```

每个 Span 记录了耗时、向量维度、检索结果数、LLM token 数等关键指标。

## 📁 项目结构

```
PrismFlow/
├── cmd/main.go                    # 入口，依赖注入
├── configs/config.yaml            # 配置文件
├── internal/
│   ├── adapter/                   # 外部依赖适配器
│   │   ├── embedding/             # llama.cpp / Ollama / Mock
│   │   ├── llm/                   # OpenAI / DeepSeek
│   │   ├── reranker/              # BGE / Mock
│   │   ├── retriever/             # HybridRetriever (RRF)
│   │   └── vectordb/              # Milvus / Redis BM25 / Mock
│   ├── api/v1/                    # HTTP Handler (chat, ingest)
│   ├── config/                    # 配置加载
│   ├── core/
│   │   ├── domain/                # 领域模型
│   │   ├── ports/                 # 接口定义
│   │   └── services/              # RAGService 编排
│   ├── infra/
│   │   ├── cache/                 # Redis 语义缓存
│   │   └── observability/         # Tracing
│   ├── middleware/                 # SemanticCache / Tracing / CORS
│   └── deployment/                # Docker Compose
├── eval/                          # 评估框架 + 测试集
├── pkg/ingest/                    # 数据入库工具（分块 + 批量 embedding）
├── docs/                          # 架构文档 + 评估文档
└── web/                           # Web UI
```

## 📄 许可证

MIT License
