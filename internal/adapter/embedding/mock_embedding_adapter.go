package embedding

import (
	"context"
	"math/rand"

	"PrismFlow/internal/core/ports"
)

type MockEmbeddingAdapter struct {
	dim int
}

// 确保实现了接口
var _ ports.EmbeddingProvider = (*MockEmbeddingAdapter)(nil)

func NewMockEmbeddingAdapter() *MockEmbeddingAdapter {
	return &MockEmbeddingAdapter{dim: 768}
}

func (m *MockEmbeddingAdapter) Embed(ctx context.Context, text string) ([]float32, error) {
	vec := make([]float32, m.dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec, nil
}

func (m *MockEmbeddingAdapter) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i := range texts {
		vec := make([]float32, m.dim)
		for j := range vec {
			vec[j] = rand.Float32()
		}
		result[i] = vec
	}
	return result, nil
}
