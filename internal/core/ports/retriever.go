package ports

import (
	"context"

	"PrismFlow/internal/core/domain"
)

// BM25Store 全文检索接口
type BM25Store interface {
	Search(ctx context.Context, query string, topK int) ([]domain.SearchResult, error)
}

// HybridRetriever 混合检索接口（向量 + BM25 融合）
type HybridRetriever interface {
	Search(ctx context.Context, query string, vector []float32, topK int) ([]domain.SearchResult, error)
}
