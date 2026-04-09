package vectordb

import (
	"context"
	"fmt"
	"log"
	"strings"

	"PrismFlow/internal/core/domain"
	"PrismFlow/internal/core/ports"

	"github.com/redis/go-redis/v9"
)

var _ ports.BM25Store = (*RedisBM25Store)(nil)

const (
	bm25IndexName = "rag_bm25_idx"
	bm25KeyPrefix = "rag:doc:"
)

// RedisBM25Store 使用 Redis Stack 的 FT.SEARCH 实现 BM25 全文检索
type RedisBM25Store struct {
	client *redis.Client
}

func NewRedisBM25Store(client *redis.Client) (*RedisBM25Store, error) {
	store := &RedisBM25Store{client: client}

	ctx := context.Background()
	if err := store.ensureBM25Index(ctx); err != nil {
		log.Printf("Warning: failed to ensure BM25 index: %v", err)
	}

	return store, nil
}

func (r *RedisBM25Store) ensureBM25Index(ctx context.Context) error {
	// 检查索引是否存在
	_, err := r.client.Do(ctx, "FT.INFO", bm25IndexName).Result()
	if err == nil {
		return nil
	}

	// 创建全文检索索引
	cmd := r.client.Do(ctx,
		"FT.CREATE", bm25IndexName,
		"ON", "HASH",
		"PREFIX", "1", bm25KeyPrefix,
		"SCHEMA",
		"content", "TEXT", "WEIGHT", "1.0",
		"title", "TEXT", "WEIGHT", "0.5",
	)

	if cmd.Err() != nil {
		return fmt.Errorf("failed to create BM25 index: %w", cmd.Err())
	}

	log.Printf("Redis BM25 index '%s' created successfully", bm25IndexName)
	return nil
}

func (r *RedisBM25Store) Search(ctx context.Context, query string, topK int) ([]domain.SearchResult, error) {
	// 对查询进行转义，避免 RediSearch 语法冲突
	escaped := escapeRedisQuery(query)

	cmd := r.client.Do(ctx,
		"FT.SEARCH", bm25IndexName,
		escaped,
		"LIMIT", "0", fmt.Sprintf("%d", topK),
		"RETURN", "2", "content", "title",
	)

	if cmd.Err() != nil {
		return nil, fmt.Errorf("BM25 search failed: %w", cmd.Err())
	}

	res, err := cmd.Result()
	if err != nil {
		return nil, err
	}

	return parseBM25Results(res)
}

// StoreDocument 将文档存入 Redis 以支持 BM25 检索
func (r *RedisBM25Store) StoreDocument(ctx context.Context, id string, content string, title string) error {
	key := fmt.Sprintf("%s%s", bm25KeyPrefix, id)
	return r.client.HSet(ctx, key, map[string]interface{}{
		"content": content,
		"title":   title,
	}).Err()
}

func parseBM25Results(res interface{}) ([]domain.SearchResult, error) {
	// 处理 map 格式（新版 go-redis）
	if resMap, ok := res.(map[interface{}]interface{}); ok {
		return parseBM25MapResult(resMap)
	}

	// 处理数组格式（旧版）
	vals, ok := res.([]interface{})
	if !ok || len(vals) < 1 {
		return nil, nil
	}

	count, ok := vals[0].(int64)
	if !ok || count == 0 {
		return nil, nil
	}

	var results []domain.SearchResult
	// 每个结果占 2 个位置：key, fields
	for i := 1; i < len(vals)-1; i += 2 {
		fields, ok := vals[i+1].([]interface{})
		if !ok {
			continue
		}

		var content, title string
		for j := 0; j < len(fields)-1; j += 2 {
			fieldName, _ := fields[j].(string)
			switch fieldName {
			case "content":
				content, _ = fields[j+1].(string)
			case "title":
				title, _ = fields[j+1].(string)
			}
		}

		results = append(results, domain.SearchResult{
			ID:      fmt.Sprintf("bm25_%d", i/2),
			Content: content,
			Score:   1.0, // BM25 分数由 RRF 融合时重新计算
			Meta:    map[string]interface{}{"title": title, "source": "bm25"},
		})
	}

	return results, nil
}

func parseBM25MapResult(resMap map[interface{}]interface{}) ([]domain.SearchResult, error) {
	totalResults, _ := resMap["total_results"].(int64)
	if totalResults == 0 {
		return nil, nil
	}

	results, ok := resMap["results"].([]interface{})
	if !ok || len(results) == 0 {
		return nil, nil
	}

	var searchResults []domain.SearchResult
	for i, r := range results {
		result, ok := r.(map[interface{}]interface{})
		if !ok {
			continue
		}

		extraAttrs, ok := result["extra_attributes"].(map[interface{}]interface{})
		if !ok {
			continue
		}

		content, _ := extraAttrs["content"].(string)
		title, _ := extraAttrs["title"].(string)

		searchResults = append(searchResults, domain.SearchResult{
			ID:      fmt.Sprintf("bm25_%d", i),
			Content: content,
			Score:   1.0,
			Meta:    map[string]interface{}{"title": title, "source": "bm25"},
		})
	}

	return searchResults, nil
}

// escapeRedisQuery 转义 RediSearch 特殊字符
func escapeRedisQuery(query string) string {
	special := []string{",", ".", "<", ">", "{", "}", "[", "]", "\"", "'", ":", ";", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "=", "~"}
	result := query
	for _, ch := range special {
		result = strings.ReplaceAll(result, ch, "\\"+ch)
	}
	return result
}
