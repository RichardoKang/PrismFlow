package embed

import (
	"context"
	"fmt"

	"PrismFlow/internal/core/ports"

	"github.com/sashabaranov/go-openai"
)

type EmbeddingAdapter struct {
	client *openai.Client
}

// 确保实现了接口
var _ ports.EmbeddingProvider = (*EmbeddingAdapter)(nil)

func NewEmbeddingAdapter(apiKey string) *EmbeddingAdapter {
	// 如果是本地模型或者其他兼容 OpenAI 协议的 endpoint，可以在这里配置 Config
	config := openai.DefaultConfig(apiKey)
	// config.BaseURL = "https://api.your-proxy.com/v1"
	return &EmbeddingAdapter{
		client: openai.NewClientWithConfig(config),
	}
}

func (a *EmbeddingAdapter) Embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := a.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: []string{text},
		Model: openai.SmallEmbedding3, // 或者 text-embedding-ada-002
	})
	if err != nil {
		return nil, err
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data returned")
	}
	// 返回第一个输入的向量
	return resp.Data[0].Embedding, nil
}
