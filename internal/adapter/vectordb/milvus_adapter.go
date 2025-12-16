// internal/adapter/vectordb/milvus_adapter.go
package vectordb

import (
	"context"
	"fmt"
	"log"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type MilvusAdapter struct {
	client         client.Client
	collectionName string
	dimension      int
}

type MilvusConfig struct {
	Address        string // localhost:19530
	CollectionName string
	Dimension      int // 1536 for OpenAI ada-002
}

func NewsMilvusAdapter(cfg MilvusConfig) (*MilvusAdapter, error) {
	ctx := context.Background()

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

		// 创建向量索引，nlist_size 可以根据数据量调整，代表聚类中心数量
		idx, _ := entity.NewIndexIvfFlat(entity.COSINE, 128)
		if err := m.client.CreateIndex(ctx, m.collectionName, "embedding", idx, false); err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}

		log.Printf("Created Milvus collection: %s", m.collectionName)
	}

	// 加载 collection 到内存
	return m.client.LoadCollection(ctx, m.collectionName, false)
}

// Search 向量检索
func (m *MilvusAdapter) Search(ctx context.Context, vector []float32, topK int) ([]SearchResult, error) {
	// 设置搜索参数，nprobes 代表搜索时扫描的簇数量，值越大精度越高但速度越慢
	sp, _ := entity.NewIndexIvfFlatSearchParam(16)

	// 执行搜索，返回 content 和 metadata 字段
	results, err := m.client.Search(
		ctx,
		m.collectionName,
		nil,
		"",
		[]string{"content", "metadata"},
		[]entity.Vector{entity.FloatVector(vector)}, // 查询向量，注意包装成 entity.Vector 切片
		"embedding",
		entity.COSINE,
		topK, // topK，返回最相似的 K 个结果
		sp,   // 搜索参数
	)
	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	var searchResults []SearchResult
	for _, result := range results {
		for i := 0; i < result.ResultCount; i++ {
			// 提取 content 和 metadata 字段
			content, _ := result.Fields.GetColumn("content").GetAsString(i)
			metadata, _ := result.Fields.GetColumn("metadata").GetAsString(i)

			searchResults = append(searchResults, SearchResult{
				Content:  content,
				Metadata: metadata,
				Score:    result.Scores[i],
			})
		}
	}
	// 返回搜索结果
	return searchResults, nil
}

// Insert 插入文档
func (m *MilvusAdapter) Insert(ctx context.Context, contents []string, embeddings [][]float32, metadata []string) error {
	contentCol := entity.NewColumnVarChar("content", contents)
	metadataCol := entity.NewColumnVarChar("metadata", metadata)
	embeddingCol := entity.NewColumnFloatVector("embedding", m.dimension, embeddings)

	_, err := m.client.Insert(ctx, m.collectionName, "", contentCol, metadataCol, embeddingCol)
	if err != nil {
		return fmt.Errorf("insert failed: %w", err)
	}

	return m.client.Flush(ctx, m.collectionName, false)
}

func (m *MilvusAdapter) Close() error {
	return m.client.Close()
}

type SearchResult struct {
	Content  string
	Metadata string
	Score    float32
}
