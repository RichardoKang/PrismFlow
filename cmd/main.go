package main

import (
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"PrismFlow/internal/adapter/llm"
	"PrismFlow/internal/adapter/vectordb"
	"PrismFlow/internal/api/middleware"
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
	var llmAdapter *llm.OpenAIAdapter
	if cfg.LLM.Provider == "deepseek" {
		llmAdapter = llm.NewDeepSeekAdapter(cfg.LLM.APIKey, cfg.LLM.Model)
	} else {
		llmAdapter = llm.NewOpenAIAdapter(cfg.LLM.APIKey, cfg.LLM.Model)
	}

	// 使用 Mock 的 VectorDB
	milvusAdapter := vectordb.NewMilvusAdapter(cfg.VectorDBAddr())

	// 使用Mock的 Embedding Adapter
	embedAdapter := llm.NewMockEmbeddingAdapter()

	// 5. 初始化 Caching
	redisCache := cache.NewRedisSemanticCache(cfg.Redis.Addr, cfg.Redis.Password)

	// 6. 初始化 Service
	ragService := services.NewRAGService(llmAdapter, milvusAdapter, embedAdapter, logger)

	// 7. 初始化 Handler
	chatHandler := v1.NewChatHandler(ragService, logger)

	// 8. 启动 HTTP Server
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
	chatGroup.Use(middleware.SemanticCacheMiddleware(embedAdapter, redisCache, logger))
	chatGroup.POST("/chat", chatHandler.HandleChat)

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
