package reranker

import (
	"context"

	"PrismFlow/internal/core/domain"
	"PrismFlow/internal/core/ports"
)

var _ ports.RerankerProvider = (*MockReranker)(nil)

// MockReranker 直接返回原始结果，用于开发测试
type MockReranker struct{}

func NewMockReranker() *MockReranker {
	return &MockReranker{}
}

func (m *MockReranker) Rerank(_ context.Context, _ string, docs []domain.SearchResult, topN int) ([]domain.SearchResult, error) {
	if len(docs) <= topN {
		return docs, nil
	}
	return docs[:topN], nil
}
