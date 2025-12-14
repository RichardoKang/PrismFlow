package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"PrismFlow/internal/infra/observability"

	"go.opentelemetry.io/otel/attribute"
)

type OpenAIEmbeddingAdapter struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

type OpenAIEmbeddingConfig struct {
	APIKey  string
	BaseURL string // 默认 https://api.openai.com/v1
	Model   string // text-embedding-ada-002 或 text-embedding-3-small
}

func NewOpenAIEmbeddingAdapter(cfg OpenAIEmbeddingConfig) *OpenAIEmbeddingAdapter {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	model := cfg.Model
	if model == "" {
		model = "text-embedding-ada-002"
	}

	return &OpenAIEmbeddingAdapter{
		apiKey:  cfg.APIKey,
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

type embeddingRequest struct {
	Input          []string `json:"input"`
	Model          string   `json:"model"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// Embed 单个文本向量化
func (a *OpenAIEmbeddingAdapter) Embed(ctx context.Context, text string) ([]float32, error) {
	vectors, err := a.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// EmbedBatch 批量向量化
func (a *OpenAIEmbeddingAdapter) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	ctx, span := observability.StartSpan(ctx, "OpenAI.Embedding")
	defer span.End()

	start := time.Now()
	observability.AddSpanAttributes(span,
		attribute.Int("embedding.input_count", len(texts)),
		attribute.String("embedding.model", a.model),
	)

	reqBody := embeddingRequest{
		Input: texts,
		Model: a.model,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		observability.SetSpanError(span, err)
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+"/embeddings", bytes.NewReader(jsonData))
	if err != nil {
		observability.SetSpanError(span, err)
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+a.apiKey)

	resp, err := a.httpClient.Do(req)
	if err != nil {
		observability.SetSpanError(span, err)
		return nil, fmt.Errorf("embedding request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("embedding API error: %s", string(body))
		observability.SetSpanError(span, err)
		return nil, err
	}

	var embResp embeddingResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		observability.SetSpanError(span, err)
		return nil, err
	}

	vectors := make([][]float32, len(texts))
	for _, data := range embResp.Data {
		vectors[data.Index] = data.Embedding
	}

	duration := time.Since(start)
	observability.AddSpanAttributes(span,
		attribute.Int64("embedding.duration_ms", duration.Milliseconds()),
		attribute.Int("embedding.vector_dim", len(vectors[0])),
		attribute.Int("embedding.total_tokens", embResp.Usage.TotalTokens),
	)
	observability.SetSpanSuccess(span)

	return vectors, nil
}

// Dimension 返回向量维度
func (a *OpenAIEmbeddingAdapter) Dimension() int {
	switch a.model {
	case "text-embedding-3-large":
		return 3072
	case "text-embedding-3-small":
		return 1536
	default: // ada-002
		return 1536
	}
}
