package ports

import (
	"context"

	"PrismFlow/internal/core/domain"
)

// RerankerProvider 精排服务接口
type RerankerProvider interface {
	Rerank(ctx context.Context, query string, docs []domain.SearchResult, topN int) ([]domain.SearchResult, error)
}
