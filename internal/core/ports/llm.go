package ports

import (
	"PrismFlow/internal/core/domain"
	"context"
)

// LLMProvider 抽象大模型服务接口
type LLMProvider interface {
	// Chat 普通对话，一次性返回结果
	Chat(ctx context.Context, messages []domain.Message) (string, error)

	// ChatStream 流式对话
	ChatStream(ctx context.Context, messages []domain.Message) (<-chan string, <-chan error)
}

// VectorStore 抽象向量数据库
type VectorStore interface {
	// Search 向量检索
	Search(ctx context.Context, vector []float32, topK int) ([]domain.SearchResult, error)

	// Store 存储向量 (用于知识库入库)
	Store(ctx context.Context, id string, vector []float32, content string, meta map[string]interface{}) error
}
