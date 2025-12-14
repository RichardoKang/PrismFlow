package ports

import (
	"PrismFlow/internal/core/domain"
	"context"
)

// LLMProvider 抽象大模型服务
// 无论是接 OpenAI, Claude 还是本地 Ollama，都必须实现此接口
type LLMProvider interface {
	// Chat 普通对话，一次性返回结果
	Chat(ctx context.Context, messages []domain.Message) (string, error)

	// ChatStream 流式对话
	// outCh 用于实时推送生成的 token
	// 返回 error 仅代表连接建立失败，后续流中断通过 close(outCh) 或发送 error struct 处理
	ChatStream(ctx context.Context, messages []domain.Message) (<-chan string, <-chan error)
}

// VectorStore 抽象向量数据库
// 无论是 Milvus, Qdrant 还是 Pgvector，都必须实现此接口
type VectorStore interface {
	// Search 向量检索
	Search(ctx context.Context, vector []float32, topK int) ([]domain.SearchResult, error)

	// Store 存储向量 (用于知识库入库)
	Store(ctx context.Context, id string, vector []float32, content string, meta map[string]interface{}) error
}
