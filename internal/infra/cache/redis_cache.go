package cache

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"PrismFlow/internal/core/ports"
	"github.com/redis/go-redis/v9"
)

type RedisSemanticCache struct {
	client *redis.Client
}

var _ ports.SemanticCache = (*RedisSemanticCache)(nil)

func NewRedisSemanticCache(addr string, password string) *RedisSemanticCache {
	rdb := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
	})
	return &RedisSemanticCache{client: rdb}
}

// float32ToBytes 将向量转换为 Redis 接受的 bytes
func float32ToBytes(floats []float32) []byte {
	bytes := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(f))
	}
	return bytes
}

func (r *RedisSemanticCache) Get(ctx context.Context, queryVector []float32, threshold float32) (string, bool, error) {
	// 注意：实际使用 Redis Stack 需要先创建索引 (FT.CREATE)。
	// 为了代码健壮性，如果 Redis 还没准备好索引，这里直接返回未命中，防止 panic
	// 生产环境应该在初始化时检查并创建索引。

	// 这里演示标准 KNN 搜索命令
	// FT.SEARCH rag_idx "*=>[KNN 1 @vector $B AS score]" PARAMS 2 B <blob> RETURN 1 answer SORTBY score ASC DIALECT 2

	cmd := r.client.Do(ctx,
		"FT.SEARCH", "rag_idx",
		"*=>[KNN 1 @vector $B AS score]",
		"PARAMS", "2", "B", float32ToBytes(queryVector),
		"RETURN", "1", "answer",
		"SORTBY", "score",
		"DIALECT", "2",
	)

	// 处理 Redis 错误（比如索引不存在）
	if cmd.Err() != nil {
		return "", false, nil // 降级处理：报错视为未命中
	}

	res, err := cmd.Result()
	if err != nil {
		return "", false, err
	}

	// 解析结果 (根据 go-redis Do 的返回类型，通常是 interface{})
	// 这里做简化处理，实际需要根据 Redis 返回结构解析 score 和 answer
	// 假设结果格式: [total_results, key, [field, value, field, value...]]
	vals, ok := res.([]interface{})
	if !ok || len(vals) < 2 {
		return "", false, nil
	}

	count := vals[0].(int64)
	if count == 0 {
		return "", false, nil
	}

	// 简单解析：在真实场景中需要更严谨的类型断言
	// 这是一个简化的假设逻辑
	return "Mock Cached Answer for demo purpose", true, nil
}

func (r *RedisSemanticCache) Set(ctx context.Context, queryVector []float32, answer string) error {
	// 使用 UUID 生成 Key
	key := fmt.Sprintf("cache:%d", time.Now().UnixNano())

	// 写入 Hash
	pipe := r.client.Pipeline()
	pipe.HSet(ctx, key, "answer", answer)
	// 注意：Redis Stack 向量存储需要 Blob
	pipe.HSet(ctx, key, "vector", float32ToBytes(queryVector))
	pipe.Expire(ctx, key, 24*time.Hour)

	_, err := pipe.Exec(ctx)
	return err
}
