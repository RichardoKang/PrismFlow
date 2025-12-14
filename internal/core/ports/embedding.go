package ports

import "context"

type EmbeddingProvider interface {
	// Embed 将文本转换为向量
	Embed(ctx context.Context, text string) ([]float32, error)
}
