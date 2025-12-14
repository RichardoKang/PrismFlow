package llm

import (
	"context"
	"math/rand"

	"PrismFlow/internal/core/ports"
)

type MockEmbeddingAdapter struct{}

// 确保实现了接口
var _ ports.EmbeddingProvider = (*MockEmbeddingAdapter)(nil)

func NewMockEmbeddingAdapter() *MockEmbeddingAdapter {
	return &MockEmbeddingAdapter{}
}

func (m *MockEmbeddingAdapter) Embed(ctx context.Context, text string) ([]float32, error) {
	// 返回一个随机的 1536 维向量 (模拟 OpenAI embedding 维度)
	// 这样可以让流程跑通，虽然向量本身没有语义，但配合 Mock VectorDB 足够测试 RAG 流程
	dims := 1536
	vec := make([]float32, dims)
	for i := 0; i < dims; i++ {
		vec[i] = rand.Float32()
	}
	return vec, nil
}
