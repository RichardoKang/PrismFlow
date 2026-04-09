package v1

import (
	"context"
	"net/http"

	"PrismFlow/internal/core/ports"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// IngestRequest 文档入库请求
type IngestRequest struct {
	Documents []IngestDocument `json:"documents" binding:"required,min=1"`
}

// IngestDocument 单个文档
type IngestDocument struct {
	Content string            `json:"content" binding:"required"`
	Meta    map[string]interface{} `json:"meta,omitempty"`
}

// IngestHandler 文档入库处理器
type IngestHandler struct {
	embedder ports.EmbeddingProvider
	vectorDB ports.VectorStore
	logger   *zap.Logger
}

func NewIngestHandler(embedder ports.EmbeddingProvider, vectorDB ports.VectorStore, logger *zap.Logger) *IngestHandler {
	return &IngestHandler{
		embedder: embedder,
		vectorDB: vectorDB,
		logger:   logger,
	}
}

// HandleIngest 处理文档入库请求
func (h *IngestHandler) HandleIngest(c *gin.Context) {
	var req IngestRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.Warn("Invalid ingest request", zap.Error(err))
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 尝试使用批量 embedding
	batchEmbedder, hasBatch := h.embedder.(interface {
		EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
	})

	ctx := c.Request.Context()

	if hasBatch {
		// 批量处理
		texts := make([]string, len(req.Documents))
		for i, doc := range req.Documents {
			texts[i] = doc.Content
		}

		vectors, err := batchEmbedder.EmbedBatch(ctx, texts)
		if err != nil {
			h.logger.Error("Batch embedding failed", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "embedding failed"})
			return
		}

		contents := make([]string, len(req.Documents))
		metas := make([]map[string]interface{}, len(req.Documents))
		for i, doc := range req.Documents {
			contents[i] = doc.Content
			metas[i] = doc.Meta
			if metas[i] == nil {
				metas[i] = map[string]interface{}{}
			}
		}

		if err := h.vectorDB.StoreBatch(ctx, vectors, contents, metas); err != nil {
			h.logger.Error("Batch store failed", zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "store failed"})
			return
		}
	} else {
		// 逐条处理
		for i, doc := range req.Documents {
			vector, err := h.embedder.Embed(ctx, doc.Content)
			if err != nil {
				h.logger.Error("Embedding failed", zap.Int("index", i), zap.Error(err))
				continue
			}
			meta := doc.Meta
			if meta == nil {
				meta = map[string]interface{}{}
			}
			if err := h.vectorDB.Store(ctx, "", vector, doc.Content, meta); err != nil {
				h.logger.Error("Store failed", zap.Int("index", i), zap.Error(err))
			}
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"count":  len(req.Documents),
	})
}
