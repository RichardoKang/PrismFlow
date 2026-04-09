// internal/adapter/vectordb/milvus_adapter.go
package vectordb

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"PrismFlow/internal/core/domain"
	"PrismFlow/internal/core/ports"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// 确保 MilvusAdapter 实现 ports.VectorStore 接口
var _ ports.VectorStore = (*MilvusAdapter)(nil)

type MilvusAdapter struct {
	client         client.Client
	collectionName string
	dimension      int
}

type MilvusConfig struct {
	Address        string // localhost:19530
	CollectionName string
	Dimension      int // 1536 for OpenAI ada-002, 768 for nomic-embedding-text
}

func NewMilvusAdapter(cfg MilvusConfig) (*MilvusAdapter, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	c, err := client.NewClient(ctx, client.Config{
		Address: cfg.Address,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to milvus: %w", err)
	}

	adapter := &MilvusAdapter{
		client:         c,
		collectionName: cfg.CollectionName,
		dimension:      cfg.Dimension,
	}

	// 确保 collection 存在
	if err := adapter.ensureCollection(ctx); err != nil {
		return nil, err
	}

	return adapter, nil
}

// ensureCollection 确保 collection 存在，如果不存在则创建
func (m *MilvusAdapter) ensureCollection(ctx context.Context) error {
	exists, err := m.client.HasCollection(ctx, m.collectionName)
	if err != nil {
		return err
	}

	if !exists {
		schema := &entity.Schema{
			CollectionName: m.collectionName,
			Fields: []*entity.Field{
				{Name: "id", DataType: entity.FieldTypeInt64, PrimaryKey: true, AutoID: true},
				{Name: "content", DataType: entity.FieldTypeVarChar, TypeParams: map[string]string{"max_length": "65535"}},
				{Name: "embedding", DataType: entity.FieldTypeFloatVector, TypeParams: map[string]string{"dim": fmt.Sprintf("%d", m.dimension)}},
				{Name: "metadata", DataType: entity.FieldTypeVarChar, TypeParams: map[string]string{"max_length": "4096"}},
			},
		}

		if err := m.client.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}

		// 创建 HNSW 向量索引，对中小数据集效果更好
		idx, _ := entity.NewIndexHNSW(entity.COSINE, 16, 256)
		if err := m.client.CreateIndex(ctx, m.collectionName, "embedding", idx, false); err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}

		log.Printf("Created Milvus collection: %s", m.collectionName)
	}

	// 加载 collection 到内存（同步等待加载完成）
	log.Printf("Loading collection '%s' into memory...", m.collectionName)
	return m.client.LoadCollection(ctx, m.collectionName, false)
}

// Search 向量检索，实现 ports.VectorStore 接口
func (m *MilvusAdapter) Search(ctx context.Context, vector []float32, topK int) ([]domain.SearchResult, error) {
	// 设置搜索参数 (HNSW ef 值，越大精度越高但速度越慢)
	sp, _ := entity.NewIndexHNSWSearchParam(64)

	// 执行搜索，返回 content 和 metadata 字段
	results, err := m.client.Search(
		ctx,
		m.collectionName,
		nil,
		"",
		[]string{"content", "metadata"},
		[]entity.Vector{entity.FloatVector(vector)},
		"embedding",
		entity.COSINE,
		topK, // topK，返回最相似的 K 个结果
		sp,   // 搜索参数
	)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	var searchResults []domain.SearchResult
	for _, result := range results {
		for i := 0; i < result.ResultCount; i++ {
			// 提取 content 和 metadata 字段
			content, _ := result.Fields.GetColumn("content").GetAsString(i)
			metadataStr, _ := result.Fields.GetColumn("metadata").GetAsString(i)

			// 解析 metadata JSON
			var meta map[string]interface{}
			if metadataStr != "" {
				_ = json.Unmarshal([]byte(metadataStr), &meta)
			}

			searchResults = append(searchResults, domain.SearchResult{
				ID:      fmt.Sprintf("%d", i),
				Content: content,
				Score:   result.Scores[i], // Milvus COSINE 原始分数
				Meta:    meta,
			})
		}
	}
	// 返回搜索结果
	return searchResults, nil
}

// Store 存储向量，实现 ports.VectorStore 接口
func (m *MilvusAdapter) Store(ctx context.Context, id string, vector []float32, content string, meta map[string]interface{}) error {
	// 将 meta 转换为 JSON 字符串
	metadataBytes, err := json.Marshal(meta)
	if err != nil {
		metadataBytes = []byte("{}")
	}
	metadataStr := string(metadataBytes)

	contentCol := entity.NewColumnVarChar("content", []string{content})
	metadataCol := entity.NewColumnVarChar("metadata", []string{metadataStr})
	embeddingCol := entity.NewColumnFloatVector("embedding", m.dimension, [][]float32{vector})

	_, err = m.client.Insert(ctx, m.collectionName, "", contentCol, metadataCol, embeddingCol)
	if err != nil {
		return fmt.Errorf("insert failed: %w", err)
	}

	return nil
}

// StoreBatch 批量存储向量，写入后统一 Flush
func (m *MilvusAdapter) StoreBatch(ctx context.Context, vectors [][]float32, contents []string, metas []map[string]interface{}) error {
	if len(vectors) == 0 {
		return nil
	}

	metaStrs := make([]string, len(metas))
	for i, meta := range metas {
		b, err := json.Marshal(meta)
		if err != nil {
			b = []byte("{}")
		}
		metaStrs[i] = string(b)
	}

	contentCol := entity.NewColumnVarChar("content", contents)
	metadataCol := entity.NewColumnVarChar("metadata", metaStrs)
	embeddingCol := entity.NewColumnFloatVector("embedding", m.dimension, vectors)

	_, err := m.client.Insert(ctx, m.collectionName, "", contentCol, metadataCol, embeddingCol)
	if err != nil {
		return fmt.Errorf("batch insert failed: %w", err)
	}

	return m.client.Flush(ctx, m.collectionName, false)
}

func (m *MilvusAdapter) Close() error {
	return m.client.Close()
}
