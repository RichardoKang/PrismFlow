package cache

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"strconv"
	"time"

	"PrismFlow/internal/core/ports"

	"github.com/redis/go-redis/v9"
)

const (
	// Redis 索引和 key 前缀
	indexName = "rag_cache_idx"
	keyPrefix = "rag:cache:"
)

type RedisSemanticCache struct {
	client    *redis.Client
	dimension int
}

var _ ports.SemanticCache = (*RedisSemanticCache)(nil)

// NewRedisSemanticCache 创建 Redis 语义缓存实例
func NewRedisSemanticCache(addr string, password string, dimension int) (*RedisSemanticCache, error) {
	rdb := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       0,
	})

	// 检查连接
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	cache := &RedisSemanticCache{
		client:    rdb,
		dimension: dimension,
	}

	// 初始化向量索引
	if err := cache.ensureIndex(ctx); err != nil {
		log.Printf("Warning: failed to ensure Redis index: %v (semantic cache may not work)", err)
	}

	// 测试 FT.SEARCH 是否可用
	testCtx, testCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer testCancel()
	testCmd := rdb.Do(testCtx, "FT.SEARCH", indexName, "*", "LIMIT", "0", "0")
	if testCmd.Err() != nil {
		log.Printf("Warning: FT.SEARCH test failed: %v", testCmd.Err())
	} else {
		log.Printf("FT.SEARCH test passed: %v", testCmd.Val())
	}

	return cache, nil
}

// ensureIndex 确保 Redis Stack 向量索引已创建
func (r *RedisSemanticCache) ensureIndex(ctx context.Context) error {
	// 检查索引是否存在
	res, err := r.client.Do(ctx, "FT.INFO", indexName).Result()
	if err == nil {
		// 索引已存在
		log.Printf("Redis index '%s' already exists", indexName)
		return nil
	}
	log.Printf("Redis FT.INFO error (will try to create index): %v", err)

	// 创建索引
	// FT.CREATE rag_cache_idx ON HASH PREFIX 1 rag:cache: SCHEMA vector VECTOR FLAT 6 TYPE FLOAT32 DIM 768 DISTANCE_METRIC COSINE answer TEXT
	cmd := r.client.Do(ctx,
		"FT.CREATE", indexName,
		"ON", "HASH",
		"PREFIX", "1", keyPrefix,
		"SCHEMA",
		"vector", "VECTOR", "FLAT", "6",
		"TYPE", "FLOAT32",
		"DIM", r.dimension,
		"DISTANCE_METRIC", "COSINE",
		"answer", "TEXT",
	)

	if cmd.Err() != nil {
		log.Printf("Failed to create Redis index: %v", cmd.Err())
		return fmt.Errorf("failed to create index: %w", cmd.Err())
	}

	log.Printf("Redis semantic cache index '%s' created successfully", indexName)

	// 验证索引是否创建成功
	res, err = r.client.Do(ctx, "FT.INFO", indexName).Result()
	if err != nil {
		log.Printf("Warning: index created but FT.INFO still fails: %v", err)
	} else {
		log.Printf("Index verified: %v", res)
	}

	return nil
}

// Ping 检查 Redis 连接是否正常
func (r *RedisSemanticCache) Ping(ctx context.Context) error {
	return r.client.Ping(ctx).Err()
}

// Close 关闭 Redis 连接
func (r *RedisSemanticCache) Close() error {
	return r.client.Close()
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
	log.Printf("Redis cache Get called: vector len=%d, threshold=%.4f", len(queryVector), threshold)

	// 使用独立的超时 context，避免外部 context 超时影响
	searchCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 使用 Redis Stack 的 FT.SEARCH 进行向量相似度搜索
	// FT.SEARCH idx "*=>[KNN 1 @vector $B AS score]" PARAMS 2 B <blob> RETURN 2 answer score SORTBY score ASC DIALECT 2

	cmd := r.client.Do(searchCtx,
		"FT.SEARCH", indexName,
		"*=>[KNN 1 @vector $B AS score]",
		"PARAMS", "2", "B", float32ToBytes(queryVector),
		"RETURN", "2", "answer", "score",
		"SORTBY", "score",
		"DIALECT", "2",
	)

	// 处理 Redis 错误（比如索引不存在）
	if cmd.Err() != nil {
		log.Printf("Redis FT.SEARCH error: %v", cmd.Err())
		return "", false, nil // 降级处理：报错视为未命中
	}

	res, err := cmd.Result()
	if err != nil {
		log.Printf("Redis Get: cmd.Result() error: %v", err)
		return "", false, err
	}

	// 新版 go-redis 返回 map 格式，需要特殊处理
	if resMap, ok := res.(map[interface{}]interface{}); ok {
		return r.parseMapResult(resMap, threshold)
	}

	// 兼容旧版数组格式
	vals, ok := res.([]interface{})
	if !ok || len(vals) < 3 {
		log.Printf("Redis Get: unexpected result type: %T", res)
		return "", false, nil
	}

	return r.parseArrayResult(vals, threshold)
}

// parseMapResult 解析新版 go-redis 返回的 map 格式结果
func (r *RedisSemanticCache) parseMapResult(resMap map[interface{}]interface{}, threshold float32) (string, bool, error) {
	// 获取 total_results
	totalResults, _ := resMap["total_results"].(int64)
	if totalResults == 0 {
		log.Printf("Redis Get: no results found")
		return "", false, nil
	}

	// 获取 results 数组
	results, ok := resMap["results"].([]interface{})
	if !ok || len(results) == 0 {
		log.Printf("Redis Get: no results in map")
		return "", false, nil
	}

	// 获取第一个结果
	firstResult, ok := results[0].(map[interface{}]interface{})
	if !ok {
		log.Printf("Redis Get: first result is not a map")
		return "", false, nil
	}

	// 获取 extra_attributes
	extraAttrs, ok := firstResult["extra_attributes"].(map[interface{}]interface{})
	if !ok {
		log.Printf("Redis Get: extra_attributes not found")
		return "", false, nil
	}

	// 提取 answer 和 score
	answer, _ := extraAttrs["answer"].(string)

	var score float64
	switch v := extraAttrs["score"].(type) {
	case float64:
		score = v
	case string:
		score, _ = strconv.ParseFloat(v, 64)
	}

	// COSINE 距离：0 表示完全相同，1 表示正交
	// 传入的 threshold 是相似度阈值（如 0.95），需要转换为距离阈值
	distanceThreshold := 1.0 - float64(threshold)
	if score > distanceThreshold {
		log.Printf("Redis cache miss: distance %.4f > threshold %.4f (similarity %.4f < %.4f)",
			score, distanceThreshold, 1.0-score, threshold)
		return "", false, nil
	}

	log.Printf("Redis cache hit: distance %.4f (similarity %.4f), answer length: %d",
		score, 1.0-score, len(answer))
	return answer, true, nil
}

// parseArrayResult 解析旧版 go-redis 返回的数组格式结果
func (r *RedisSemanticCache) parseArrayResult(vals []interface{}, threshold float32) (string, bool, error) {
	// 第一个元素是结果数量
	count, ok := vals[0].(int64)
	if !ok || count == 0 {
		log.Printf("Redis Get: count=0 or type assertion failed, vals[0]=%v", vals[0])
		return "", false, nil
	}

	log.Printf("Redis Get: found %d results", count)

	// 解析文档字段 (vals[2] 是第一个文档的字段数组)
	fields, ok := vals[2].([]interface{})
	if !ok || len(fields) < 4 {
		log.Printf("Redis Get: fields type assertion failed or len < 4, vals[2]=%v", vals[2])
		return "", false, nil
	}

	// 解析 answer 和 score
	var answer string
	var score float64

	for i := 0; i < len(fields)-1; i += 2 {
		fieldName, ok := fields[i].(string)
		if !ok {
			continue
		}
		switch fieldName {
		case "answer":
			if v, ok := fields[i+1].(string); ok {
				answer = v
			}
		case "score":
			if v, ok := fields[i+1].(string); ok {
				score, _ = strconv.ParseFloat(v, 64)
			}
		}
	}

	// COSINE 距离：0 表示完全相同，1 表示正交
	distanceThreshold := 1.0 - float64(threshold)
	if score > distanceThreshold {
		log.Printf("Redis cache miss: distance %.4f > threshold %.4f (similarity %.4f < %.4f)",
			score, distanceThreshold, 1.0-score, threshold)
		return "", false, nil
	}

	log.Printf("Redis cache hit: distance %.4f (similarity %.4f), answer length: %d",
		score, 1.0-score, len(answer))
	return answer, true, nil
}

func (r *RedisSemanticCache) Set(ctx context.Context, queryVector []float32, answer string) error {
	// 使用正确的 key 前缀，确保被索引捕获
	key := fmt.Sprintf("%s%d", keyPrefix, time.Now().UnixNano())

	// 写入 Hash，确保字段名与索引定义一致
	pipe := r.client.Pipeline()
	pipe.HSet(ctx, key, "answer", answer)
	// Redis Stack 向量存储需要 Blob 格式
	pipe.HSet(ctx, key, "vector", float32ToBytes(queryVector))
	pipe.Expire(ctx, key, 24*time.Hour)

	_, err := pipe.Exec(ctx)
	if err != nil {
		log.Printf("Redis cache set error: %v", err)
		return err
	}

	log.Printf("Redis cache set: key=%s, answer length=%d", key, len(answer))
	return nil
}
