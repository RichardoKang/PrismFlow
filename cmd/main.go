package main

import (
	"PrismFlow/internal/adapter/embedding"
	"PrismFlow/internal/adapter/reranker"
	"PrismFlow/internal/adapter/retriever"
	"PrismFlow/internal/core/ports"
	"PrismFlow/internal/middleware"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"PrismFlow/internal/adapter/llm"
	"PrismFlow/internal/adapter/vectordb"
	v1 "PrismFlow/internal/api/v1"
	"PrismFlow/internal/config"
	"PrismFlow/internal/core/services"
	"PrismFlow/internal/infra/cache"
	"PrismFlow/internal/infra/observability"
)

func main() {
	// 1. 加载配置
	cfg, err := config.Load("configs/config.yaml")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// 2. 初始化 Logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// 3. 初始化 Tracing
	shutdownTracer := observability.InitTracer(cfg.Server.Name, cfg.Trace.Endpoint)
	defer func() {
		if err := shutdownTracer(nil); err != nil {
			logger.Error("failed to shutdown tracer", zap.Error(err))
		}
	}()

	// 4. 初始化 Adapters
	var llmAdapter ports.LLMProvider
	if cfg.LLM.Provider == "deepseek" {
		llmAdapter = llm.NewDeepSeekAdapter(cfg.LLM.APIKey, cfg.LLM.Model)
	} else {
		llmAdapter = llm.NewOpenAIAdapter(cfg.LLM.APIKey, cfg.LLM.Model)
	}

	// 使用真正的 Milvus 向量数据库
	milvusCfg := vectordb.MilvusConfig{
		Address:        cfg.VectorDBAddr(),
		CollectionName: cfg.VectorDB.CollectionName,
		Dimension:      cfg.VectorDB.Dimension,
	}
	milvusAdapter, err := vectordb.NewMilvusAdapter(milvusCfg)
	if err != nil {
		log.Fatalf("Failed to connect to Milvus: %v", err)
	}
	defer milvusAdapter.Close()

	// 根据配置选择 Embedding Adapter
	var embedAdapter ports.EmbeddingProvider
	switch cfg.Embedding.Provider {
	case "ollama":
		embedAdapter = embedding.NewOllamaEmbeddingAdapter(embedding.OllamaConfig{
			BaseURL: cfg.Embedding.BaseURL,
			Model:   cfg.Embedding.Model,
		})
		logger.Info("Using Ollama embedding adapter",
			zap.String("base_url", cfg.Embedding.BaseURL),
			zap.String("model", cfg.Embedding.Model),
		)
	case "llamacpp":
		embedAdapter = embedding.NewLlamaCppEmbeddingAdapter(cfg.Embedding.BaseURL)
		logger.Info("Using llama.cpp embedding adapter",
			zap.String("base_url", cfg.Embedding.BaseURL),
		)
	default:
		embedAdapter = embedding.NewMockEmbeddingAdapter()
		logger.Info("Using Mock embedding adapter")
	}

	// 5. 初始化 Caching
	redisCache, err := cache.NewRedisSemanticCache(cfg.Redis.Addr, cfg.Redis.Password, cfg.VectorDB.Dimension)
	if err != nil {
		logger.Warn("Failed to initialize Redis cache, semantic cache disabled", zap.Error(err))
		redisCache = nil
	}

	// 6. 初始化 Reranker
	var rerankerAdapter ports.RerankerProvider
	if cfg.Reranker.Provider == "bge" {
		rerankerAdapter = reranker.NewBGEReranker(reranker.BGEConfig{
			BaseURL: cfg.Reranker.BaseURL,
			Model:   cfg.Reranker.Model,
		}, logger)
		logger.Info("Using BGE reranker",
			zap.String("base_url", cfg.Reranker.BaseURL),
			zap.String("model", cfg.Reranker.Model),
		)
	} else {
		rerankerAdapter = reranker.NewMockReranker()
		logger.Info("Using Mock reranker")
	}

	// 7. 初始化 Hybrid Retriever（BM25 + Vector，使用 Redis Stack 做全文检索）
	var hybridRetriever ports.HybridRetriever
	if redisCache != nil {
		redisBM25, bm25Err := vectordb.NewRedisBM25Store(redisCache.Client())
		if bm25Err != nil {
			logger.Warn("Failed to initialize BM25 store, hybrid search disabled", zap.Error(bm25Err))
			hybridRetriever = retriever.NewHybridRetriever(milvusAdapter, nil, logger)
		} else {
			hybridRetriever = retriever.NewHybridRetriever(milvusAdapter, redisBM25, logger)
			logger.Info("Hybrid search enabled (Vector + BM25)")
		}
	} else {
		hybridRetriever = retriever.NewHybridRetriever(milvusAdapter, nil, logger)
		logger.Info("Hybrid search disabled (no Redis), using vector-only retrieval")
	}

	// 8. 初始化 Service
	ragService := services.NewRAGService(llmAdapter, milvusAdapter, embedAdapter, rerankerAdapter, hybridRetriever, logger, cfg.RAG.ScoreThreshold, cfg.RAG.TopK, cfg.Reranker.TopN)

	// 9. 初始化 Handler
	chatHandler := v1.NewChatHandler(ragService, logger)
	ingestHandler := v1.NewIngestHandler(embedAdapter, milvusAdapter, logger)

	// 10. 启动 HTTP Server
	r := gin.New()
	r.Use(gin.Recovery())

	// CORS 中间件 - 允许前端跨域访问
	r.Use(corsMiddleware())

	// 植入中间件
	r.Use(middleware.TracingMiddleware(cfg.Server.Name))

	// 静态文件服务 - 提供 Web UI
	r.StaticFile("/", "./web/index.html")
	r.StaticFile("/index.html", "./web/index.html")
	r.Static("/static", "./web/static")

	// API 路由 (语义缓存只对 chat 接口生效)
	chatGroup := r.Group("/v1")
	chatGroup.Use(middleware.SemanticCacheMiddleware(embedAdapter, redisCache, logger, cfg.RAG.CacheThreshold))
	chatGroup.POST("/chat", chatHandler.HandleChat)

	// Ingest API（无需语义缓存）
	r.POST("/v1/ingest", ingestHandler.HandleIngest)

	// 健康检查接口
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "ok",
			"service": cfg.Server.Name,
			"time":    time.Now().Format(time.RFC3339),
		})
	})

	logger.Info("RAG Gateway starting",
		zap.Int("port", cfg.Server.Port),
		zap.String("web_ui", "http://localhost:8080"),
	)
	if err := r.Run(":8080"); err != nil {
		log.Fatal(err)
	}
}

// corsMiddleware 处理跨域请求
func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
		c.Header("Access-Control-Allow-Credentials", "true")
		c.Header("Access-Control-Max-Age", "86400")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}
