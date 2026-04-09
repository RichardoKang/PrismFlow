package vectordb

import (
	"context"
	"time"

	"PrismFlow/internal/core/domain"
	"PrismFlow/internal/core/ports"
)

type MockVectorAdapter struct{}

var _ ports.VectorStore = (*MockVectorAdapter)(nil)

func NewMockVectorAdapter() *MockVectorAdapter {
	return &MockVectorAdapter{}
}

func (m *MockVectorAdapter) Search(ctx context.Context, vector []float32, topK int) ([]domain.SearchResult, error) {
	// 模拟网络延迟
	select {
	case <-time.After(50 * time.Millisecond):
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// 返回假数据
	return []domain.SearchResult{
		{
			ID:      "1",
			Content: "Go 语言（Golang）是 Google 开发的一种静态强类型、编译型、并发型，并具有垃圾回收功能的编程语言。",
			Score:   0.95,
			Meta:    map[string]interface{}{"source": "wiki"},
		},
		{
			ID:      "2",
			Content: "RAG (Retrieval-Augmented Generation) 是一种结合了检索和生成的 AI 应用架构。",
			Score:   0.88,
			Meta:    map[string]interface{}{"source": "docs"},
		},
	}, nil
}

func (m *MockVectorAdapter) Store(ctx context.Context, id string, vector []float32, content string, meta map[string]interface{}) error {
	return nil
}

func (m *MockVectorAdapter) StoreBatch(ctx context.Context, vectors [][]float32, contents []string, metas []map[string]interface{}) error {
	return nil
}
