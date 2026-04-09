package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"PrismFlow/internal/core/ports"
	"PrismFlow/internal/infra/observability"

	"go.opentelemetry.io/otel/attribute"
)

var _ ports.EmbeddingProvider = (*LlamaCppEmbeddingAdapter)(nil)

type LlamaCppEmbeddingAdapter struct {
	baseURL    string
	httpClient *http.Client
}

func NewLlamaCppEmbeddingAdapter(baseURL string) *LlamaCppEmbeddingAdapter {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	return &LlamaCppEmbeddingAdapter{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

type llamaCppEmbedRequest struct {
	Input interface{} `json:"input"` // string or []string
	Model string      `json:"model"`
}

type llamaCppEmbedResponse struct {
	Data  []llamaCppEmbedData `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
	} `json:"usage"`
}

type llamaCppEmbedData struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

func (a *LlamaCppEmbeddingAdapter) Embed(ctx context.Context, text string) ([]float32, error) {
	// 单条查询用 search_query 前缀
	return a.embedWithPrefix(ctx, "search_query: "+text)
}

func (a *LlamaCppEmbeddingAdapter) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	// 批量默认用 search_document 前缀（用于文档入库）
	return a.embedBatchWithPrefix(ctx, texts, "search_document: ")
}

// EmbedBatchQuery 批量查询 embedding（用 search_query 前缀）
func (a *LlamaCppEmbeddingAdapter) EmbedBatchQuery(ctx context.Context, texts []string) ([][]float32, error) {
	return a.embedBatchWithPrefix(ctx, texts, "search_query: ")
}

func (a *LlamaCppEmbeddingAdapter) embedWithPrefix(ctx context.Context, text string) ([]float32, error) {
	vectors, err := a.embedBatchWithPrefix(ctx, []string{text}, "")
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return nil, fmt.Errorf("empty embedding returned")
	}
	return vectors[0], nil
}

func (a *LlamaCppEmbeddingAdapter) embedBatchWithPrefix(ctx context.Context, texts []string, prefix string) ([][]float32, error) {
	ctx, span := observability.StartSpan(ctx, "LlamaCpp.Embedding")
	defer span.End()

	start := time.Now()
	observability.AddSpanAttributes(span,
		attribute.Int("embedding.input_count", len(texts)),
		attribute.String("embedding.provider", "llama.cpp"),
	)

	// 添加前缀
	prefixed := make([]string, len(texts))
	for i, t := range texts {
		if prefix != "" {
			prefixed[i] = prefix + t
		} else {
			prefixed[i] = t
		}
	}

	reqBody := llamaCppEmbedRequest{
		Input: prefixed,
		Model: "nomic-embed-text",
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		observability.SetSpanError(span, err)
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+"/v1/embeddings", bytes.NewReader(jsonData))
	if err != nil {
		observability.SetSpanError(span, err)
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := a.httpClient.Do(req)
	if err != nil {
		observability.SetSpanError(span, err)
		return nil, fmt.Errorf("llama.cpp request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		observability.SetSpanError(span, fmt.Errorf("status: %d", resp.StatusCode))
		return nil, fmt.Errorf("llama.cpp API returned status: %d", resp.StatusCode)
	}

	var embedResp llamaCppEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		observability.SetSpanError(span, err)
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(embedResp.Data) == 0 {
		err := fmt.Errorf("empty embeddings returned for %d inputs", len(texts))
		observability.SetSpanError(span, err)
		return nil, err
	}

	// 按 index 排序结果
	results := make([][]float32, len(texts))
	for _, d := range embedResp.Data {
		if d.Index < len(results) {
			results[d.Index] = d.Embedding
		}
	}

	duration := time.Since(start)
	observability.AddSpanAttributes(span,
		attribute.Int64("embedding.duration_ms", duration.Milliseconds()),
		attribute.Int("embedding.vector_dim", len(results[0])),
		attribute.Int("embedding.total_tokens", embedResp.Usage.PromptTokens),
	)
	observability.SetSpanSuccess(span)

	return results, nil
}
