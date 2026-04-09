# PrismFlow 架构设计

## 系统概览

PrismFlow 是一个基于 Go 语言的 RAG (Retrieval-Augmented Generation) 学习验证项目，采用六边形架构（Ports & Adapters），核心模块与外部依赖解耦。

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        HTTP Layer                           │
│  ┌──────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ /v1/chat │  │ /v1/ingest       │  │ /health          │  │
│  └────┬─────┘  └────────┬─────────┘  └──────────────────┘  │
│       │                 │                                    │
│  ┌────▼─────────────────▼────────────────────────────────┐  │
│  │              Middleware Layer                          │  │
│  │  Tracing │ SemanticCache │ CORS                       │  │
│  └────┬──────────────────────────────────────────────────┘  │
└───────┼─────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│                     Core (Domain)                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  RAGService                           │   │
│  │                                                       │   │
│  │  1. Embedding (复用缓存中间件已计算的向量)             │   │
│  │  2. Retrieval (HybridRetriever: Vector + BM25/RRF)   │   │
│  │  3. Rerank   (BGE Reranker 精排)                     │   │
│  │  4. Prompt Assembly (过滤低分 + 构建上下文)           │   │
│  │  5. LLM Generation (流式输出)                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Ports (接口定义):                                           │
│  ├── LLMProvider        ├── EmbeddingProvider               │
│  ├── VectorStore        ├── SemanticCache                   │
│  ├── RerankerProvider   ├── HybridRetriever                 │
│  └── BM25Store                                              │
└──────────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│                    Adapters (实现层)                          │
│                                                              │
│  LLM:        OpenAI / DeepSeek (openai-compatible)          │
│  Embedding:  llama.cpp / Ollama (nomic-embed-text) / Mock   │
│  VectorDB:   Milvus (HNSW + COSINE)                        │
│  Cache:      Redis Stack (向量相似度缓存)                    │
│  BM25:       Redis Stack (FT.SEARCH 全文检索)               │
│  Reranker:   BGE-Reranker-v2-m3 (TEI/Xinference) / Mock    │
│  Tracing:    OpenTelemetry → Jaeger                         │
└──────────────────────────────────────────────────────────────┘
```

## RAG 流程

```
User Query
    │
    ▼
SemanticCache Middleware ──hit──▶ 返回缓存结果
    │ miss (传递 query_vector via gin.Context)
    ▼
RAGService.StreamChat(ctx, query, queryVector)
    │
    ├─ 1. Embedding (复用缓存中间件已计算的向量，避免双重计算)
    │
    ├─ 2. Retrieval
    │     ├── Vector Search (Milvus HNSW) ─┐
    │     └── BM25 Search (Redis FT)      ─┤── RRF Fusion
    │                                       │
    ├─ 3. Rerank (BGE Reranker)            ◄┘
    │
    ├─ 4. Prompt Assembly (过滤 + 拼接上下文)
    │
    └─ 5. LLM Stream Generation ──▶ SSE Response
                                       │
                                  异步写入缓存
```

## Embedding 前缀策略

nomic-embed-text 模型要求对查询和文档使用不同的前缀：

| 场景 | 前缀 | 说明 |
|------|------|------|
| 查询 | `search_query: ` | 用于 chat 时的用户问题 embedding |
| 文档 | `search_document: ` | 用于 ingest 时的文档 embedding |

`LlamaCppEmbeddingAdapter` 的 `Embed()` 自动添加 `search_query:` 前缀，`EmbedBatch()` 自动添加 `search_document:` 前缀。

## Ingest 流程

```
Markdown 文件
    │
    ├─ 1. 按标题 (# / ##) 分段
    │
    ├─ 2. 段落内按句子边界分块 (chunk_size + overlap)
    │     └── 去除 Markdown 格式标记
    │
    ├─ 3. 批量计算 embedding (EmbedBatch, search_document: 前缀)
    │
    └─ 4. 批量写入 Milvus (StoreBatch + 单次 Flush)
```

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 语言 | Go 1.25 |
| Web 框架 | Gin |
| LLM | DeepSeek / OpenAI Compatible |
| Embedding | llama.cpp / Ollama (nomic-embed-text, 768d) |
| 向量数据库 | Milvus 2.3+ (HNSW + COSINE) |
| 缓存 & BM25 | Redis Stack |
| 精排 | BGE-Reranker-v2-m3 |
| 链路追踪 | OpenTelemetry + Jaeger |
| 容器编排 | Docker Compose |

## 关键设计决策

### 双重 Embedding 优化
SemanticCache 中间件在缓存 miss 时，将已计算的 query vector 通过 `gin.Context.Set("query_vector", vector)` 传递给下游。RAGService 检测到已有向量时直接复用，避免重复计算。

### HNSW 索引
选择 HNSW 替代 IVF_FLAT，对中小规模数据集（< 10 万条）检索质量更好，无需调优 nlist/nprobe 参数。

### RRF 融合
Hybrid Search 使用 Reciprocal Rank Fusion (k=60) 融合向量检索和 BM25 结果，公式：`score(doc) = Σ 1/(k + rank_i)`。当 BM25 不可用时自动降级为纯向量检索。

### 配置外提
语义缓存阈值、检索分数阈值、topK 等参数从 `config.yaml` 读取，避免硬编码。

## 目录结构

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
