package ports

import (
	"context"
)

type SemanticCache interface {
	// Get 尝试获取相似问题的答案
	Get(ctx context.Context, queryVector []float32, threshold float32) (string, bool, error)

	// Set 异步存入新的问答对
	Set(ctx context.Context, queryVector []float32, answer string) error
}
