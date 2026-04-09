package retriever

import (
	"context"
	"sort"
	"sync"

	"PrismFlow/internal/core/domain"
	"PrismFlow/internal/core/ports"
	"PrismFlow/internal/infra/observability"

	"go.opentelemetry.io/otel/attribute"
	"go.uber.org/zap"
)

var _ ports.HybridRetriever = (*HybridRetriever)(nil)

// HybridRetriever 混合检索：向量检索 + BM25 全文检索，使用 RRF 融合
type HybridRetriever struct {
	vectorStore ports.VectorStore
	bm25Store   ports.BM25Store // 可为 nil，降级为纯向量检索
	logger      *zap.Logger
	rrfK        int // RRF 参数，默认 60
}

func NewHybridRetriever(vectorStore ports.VectorStore, bm25Store ports.BM25Store, logger *zap.Logger) *HybridRetriever {
	return &HybridRetriever{
		vectorStore: vectorStore,
		bm25Store:   bm25Store,
		logger:      logger,
		rrfK:        60,
	}
}

func (h *HybridRetriever) Search(ctx context.Context, query string, vector []float32, topK int) ([]domain.SearchResult, error) {
	ctx, span := observability.StartSpan(ctx, "HybridRetriever.Search")
	defer span.End()

	span.SetAttributes(
		attribute.Int("hybrid.top_k", topK),
		attribute.Bool("hybrid.bm25_enabled", h.bm25Store != nil),
	)

	// 如果没有 BM25Store，降级为纯向量检索
	if h.bm25Store == nil {
		results, err := h.vectorStore.Search(ctx, vector, topK)
		if err != nil {
			observability.SetSpanError(span, err)
			return nil, err
		}
		span.SetAttributes(attribute.String("hybrid.mode", "vector_only"))
		observability.SetSpanSuccess(span)
		return results, nil
	}

	// 并发执行两路检索
	var (
		vectorResults []domain.SearchResult
		bm25Results   []domain.SearchResult
		vectorErr     error
		bm25Err       error
		wg            sync.WaitGroup
	)

	wg.Add(2)

	go func() {
		defer wg.Done()
		vectorResults, vectorErr = h.vectorStore.Search(ctx, vector, topK)
	}()

	go func() {
		defer wg.Done()
		bm25Results, bm25Err = h.bm25Store.Search(ctx, query, topK)
	}()

	wg.Wait()

	// 任一路失败时降级到另一路
	if vectorErr != nil && bm25Err != nil {
		observability.SetSpanError(span, vectorErr)
		return nil, vectorErr
	}

	if vectorErr != nil {
		h.logger.Warn("Vector search failed, falling back to BM25 only", zap.Error(vectorErr))
		span.SetAttributes(attribute.String("hybrid.mode", "bm25_only"))
		observability.SetSpanSuccess(span)
		return bm25Results, nil
	}

	if bm25Err != nil {
		h.logger.Warn("BM25 search failed, falling back to vector only", zap.Error(bm25Err))
		span.SetAttributes(attribute.String("hybrid.mode", "vector_only"))
		observability.SetSpanSuccess(span)
		return vectorResults, nil
	}

	// RRF 融合
	fused := h.rrfFusion(vectorResults, bm25Results, topK)

	span.SetAttributes(
		attribute.String("hybrid.mode", "hybrid_rrf"),
		attribute.Int("hybrid.vector_results", len(vectorResults)),
		attribute.Int("hybrid.bm25_results", len(bm25Results)),
		attribute.Int("hybrid.fused_results", len(fused)),
	)
	observability.SetSpanSuccess(span)

	return fused, nil
}

// rrfFusion 使用 Reciprocal Rank Fusion 融合两路结果
// score(doc) = Σ 1/(k + rank_i)
func (h *HybridRetriever) rrfFusion(vectorResults, bm25Results []domain.SearchResult, topK int) []domain.SearchResult {
	type docScore struct {
		doc   domain.SearchResult
		score float64
	}

	scoreMap := make(map[string]*docScore)

	// 向量检索结果的 RRF 分数
	for rank, doc := range vectorResults {
		key := doc.Content // 用 content 作为去重 key
		if existing, ok := scoreMap[key]; ok {
			existing.score += 1.0 / float64(h.rrfK+rank+1)
		} else {
			scoreMap[key] = &docScore{
				doc:   doc,
				score: 1.0 / float64(h.rrfK+rank+1),
			}
		}
	}

	// BM25 检索结果的 RRF 分数
	for rank, doc := range bm25Results {
		key := doc.Content
		if existing, ok := scoreMap[key]; ok {
			existing.score += 1.0 / float64(h.rrfK+rank+1)
		} else {
			scoreMap[key] = &docScore{
				doc:   doc,
				score: 1.0 / float64(h.rrfK+rank+1),
			}
		}
	}

	// 转为切片并按 RRF 分数降序排序
	results := make([]docScore, 0, len(scoreMap))
	for _, ds := range scoreMap {
		results = append(results, *ds)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	// 取 topK
	if len(results) > topK {
		results = results[:topK]
	}

	// 转换回 SearchResult，用 RRF 分数替换原始分数
	output := make([]domain.SearchResult, len(results))
	for i, ds := range results {
		doc := ds.doc
		doc.Score = float32(ds.score)
		output[i] = doc
	}

	return output
}
