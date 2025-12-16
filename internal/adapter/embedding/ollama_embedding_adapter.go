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

// 确保实现了接口
var _ ports.EmbeddingProvider = (*OllamaEmbeddingAdapter)(nil)

type OllamaEmbeddingAdapter struct {
	baseURL    string
	model      string
	httpClient *http.Client
}

type OllamaConfig struct {
	BaseURL string // 默认 http://localhost:11434
	Model   string // 默认 nomic-embed-text
}

func NewOllamaEmbeddingAdapter(cfg OllamaConfig) *OllamaEmbeddingAdapter {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	model := cfg.Model
	if model == "" {
		model = "nomic-embed-text"
	}

	return &OllamaEmbeddingAdapter{
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: 60 * time.Second, // 本地推理可能比云端慢，稍微调大超时
		},
	}
}

// Ollama 的请求结构
type ollamaRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// Ollama 的响应结构
type ollamaResponse struct {
	Embeddings      [][]float32 `json:"embeddings"`
	PromptEvalCount int         `json:"prompt_eval_count"` // Token 数量
	TotalDuration   int64       `json:"total_duration"`    // 纳秒
}

func (a *OllamaEmbeddingAdapter) Embed(ctx context.Context, text string) ([]float32, error) {
	vectors, err := a.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, fmt.Errorf("embedding failed for text (len=%d): %w", len(text), err)
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("no embedding returned for text (len=%d)", len(text))
	}
	if len(vectors[0]) == 0 {
		return nil, fmt.Errorf("empty vector returned for text (len=%d)", len(text))
	}
	return vectors[0], nil
}

func (a *OllamaEmbeddingAdapter) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	ctx, span := observability.StartSpan(ctx, "Ollama.Embedding")
	defer span.End()

	start := time.Now()
	observability.AddSpanAttributes(span,
		attribute.Int("embedding.input_count", len(texts)),
		attribute.String("embedding.model", a.model),
		attribute.String("embedding.provider", "ollama"),
	)

	reqBody := ollamaRequest{
		Model: a.model,
		Input: texts,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		observability.SetSpanError(span, err)
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// 注意：Ollama 的批量接口是 /api/embed
	req, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+"/api/embed", bytes.NewReader(jsonData))
	if err != nil {
		observability.SetSpanError(span, err)
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := a.httpClient.Do(req)
	if err != nil {
		observability.SetSpanError(span, err)
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		observability.SetSpanError(span, fmt.Errorf("ollama status: %d", resp.StatusCode))
		return nil, fmt.Errorf("ollama API returned status: %d", resp.StatusCode)
	}

	var ollamaResp ollamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		observability.SetSpanError(span, err)
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// 检查返回的 embeddings 是否为空
	if len(ollamaResp.Embeddings) == 0 {
		err := fmt.Errorf("ollama returned empty embeddings for %d input(s)", len(texts))
		observability.SetSpanError(span, err)
		return nil, err
	}

	duration := time.Since(start)
	observability.AddSpanAttributes(span,
		attribute.Int64("embedding.duration_ms", duration.Milliseconds()),
		attribute.Int("embedding.vector_dim", len(ollamaResp.Embeddings[0])),
		attribute.Int("embedding.total_tokens", ollamaResp.PromptEvalCount),
	)
	observability.SetSpanSuccess(span)

	return ollamaResp.Embeddings, nil
}

func (a *OllamaEmbeddingAdapter) Dimension() int {
	// nomic-embedding-text 是 768 维
	// mxbai-embedding-large 是 1024 维
	// llama2/3 是 4096 维
	switch a.model {
	case "nomic-embedding-text":
		return 768
	case "mxbai-embedding-large":
		return 1024
	default:
		return 768 // 默认假设
	}
}
