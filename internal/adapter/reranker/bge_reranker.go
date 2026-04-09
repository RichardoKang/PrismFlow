package reranker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"time"

	"PrismFlow/internal/core/domain"
	"PrismFlow/internal/core/ports"
	"PrismFlow/internal/infra/observability"

	"go.opentelemetry.io/otel/attribute"
	"go.uber.org/zap"
)

var _ ports.RerankerProvider = (*BGEReranker)(nil)

// BGEReranker 通过 HTTP 调用本地 reranker 服务（TEI / Xinference）
type BGEReranker struct {
	baseURL    string
	model      string
	httpClient *http.Client
	logger     *zap.Logger
}

type BGEConfig struct {
	BaseURL string
	Model   string
}

func NewBGEReranker(cfg BGEConfig, logger *zap.Logger) *BGEReranker {
	return &BGEReranker{
		baseURL: cfg.BaseURL,
		model:   cfg.Model,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		logger: logger,
	}
}

// reranker 请求/响应结构（兼容 TEI 和 Xinference）
type rerankRequest struct {
	Query    string   `json:"query"`
	Texts    []string `json:"texts"`
	Model    string   `json:"model,omitempty"`
	TopN     int      `json:"top_n,omitempty"`
}

type rerankResponse struct {
	Results []rerankResult `json:"results"`
}

type rerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

func (r *BGEReranker) Rerank(ctx context.Context, query string, docs []domain.SearchResult, topN int) ([]domain.SearchResult, error) {
	if len(docs) == 0 {
		return docs, nil
	}

	ctx, span := observability.StartSpan(ctx, "Reranker.BGE")
	defer span.End()

	span.SetAttributes(
		attribute.String("reranker.model", r.model),
		attribute.Int("reranker.input_docs", len(docs)),
		attribute.Int("reranker.top_n", topN),
	)

	// 构建请求
	texts := make([]string, len(docs))
	for i, doc := range docs {
		texts[i] = doc.Content
	}

	reqBody := rerankRequest{
		Query: query,
		Texts: texts,
		Model: r.model,
		TopN:  topN,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		observability.SetSpanError(span, err)
		return docs, nil // 降级返回原始结果
	}

	req, err := http.NewRequestWithContext(ctx, "POST", r.baseURL+"/rerank", bytes.NewReader(jsonData))
	if err != nil {
		observability.SetSpanError(span, err)
		return docs, nil
	}
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := r.httpClient.Do(req)
	duration := time.Since(start)

	span.SetAttributes(attribute.Float64("reranker.duration_ms", float64(duration.Milliseconds())))

	if err != nil {
		observability.SetSpanError(span, err)
		r.logger.Warn("Reranker request failed, falling back to original order", zap.Error(err))
		return docs, nil // 降级
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("reranker returned status %d", resp.StatusCode)
		observability.SetSpanError(span, err)
		r.logger.Warn("Reranker returned non-200", zap.Int("status", resp.StatusCode))
		return docs, nil // 降级
	}

	var rerankResp rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&rerankResp); err != nil {
		observability.SetSpanError(span, err)
		r.logger.Warn("Failed to decode reranker response", zap.Error(err))
		return docs, nil
	}

	// 按 relevance_score 降序排序
	sort.Slice(rerankResp.Results, func(i, j int) bool {
		return rerankResp.Results[i].RelevanceScore > rerankResp.Results[j].RelevanceScore
	})

	// 构建重排后的结果
	reranked := make([]domain.SearchResult, 0, topN)
	for i, result := range rerankResp.Results {
		if i >= topN {
			break
		}
		if result.Index < len(docs) {
			doc := docs[result.Index]
			doc.Score = float32(result.RelevanceScore)
			reranked = append(reranked, doc)
		}
	}

	span.SetAttributes(attribute.Int("reranker.output_docs", len(reranked)))
	observability.SetSpanSuccess(span)

	return reranked, nil
}
