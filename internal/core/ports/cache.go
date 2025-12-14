package ports

import (
	"context"
)

type SemanticCache interface {
	// Get 尝试获取相似问题的答案
	// queryVector: 当前问题的向量
	// threshold: 相似度阈值 (e.g., 0.95)
	// 返回: (cachedAnswer, hit, error)
	Get(ctx context.Context, queryVector []float32, threshold float32) (string, bool, error)

	// Set 异步存入新的问答对
	Set(ctx context.Context, queryVector []float32, answer string) error
}
