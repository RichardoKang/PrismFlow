package mocks

import (
	"context"

	"PrismFlow/internal/core/domain"
)

// MockLLMProvider 模拟 LLM 提供者
type MockLLMProvider struct {
	ChatFunc       func(ctx context.Context, messages []domain.Message) (string, error)
	ChatStreamFunc func(ctx context.Context, messages []domain.Message) (<-chan string, <-chan error)
}

func (m *MockLLMProvider) Chat(ctx context.Context, messages []domain.Message) (string, error) {
	if m.ChatFunc != nil {
		return m.ChatFunc(ctx, messages)
	}
	return "mock response", nil
}

func (m *MockLLMProvider) ChatStream(ctx context.Context, messages []domain.Message) (<-chan string, <-chan error) {
	if m.ChatStreamFunc != nil {
		return m.ChatStreamFunc(ctx, messages)
	}
	tokenChan := make(chan string)
	errChan := make(chan error, 1)
	go func() {
		defer close(tokenChan)
		defer close(errChan)
		tokenChan <- "Hello"
		tokenChan <- " World"
	}()
	return tokenChan, errChan
}

// MockVectorStore 模拟向量存储
type MockVectorStore struct {
	SearchFunc func(ctx context.Context, vector []float32, topK int) ([]domain.SearchResult, error)
	StoreFunc  func(ctx context.Context, id string, vector []float32, content string, meta map[string]interface{}) error
}

func (m *MockVectorStore) Search(ctx context.Context, vector []float32, topK int) ([]domain.SearchResult, error) {
	if m.SearchFunc != nil {
		return m.SearchFunc(ctx, vector, topK)
	}
	return []domain.SearchResult{
		{ID: "1", Content: "Test content", Score: 0.95},
	}, nil
}

func (m *MockVectorStore) Store(ctx context.Context, id string, vector []float32, content string, meta map[string]interface{}) error {
	if m.StoreFunc != nil {
		return m.StoreFunc(ctx, id, vector, content, meta)
	}
	return nil
}

// MockEmbeddingProvider 模拟 Embedding 提供者
type MockEmbeddingProvider struct {
	EmbedFunc func(ctx context.Context, text string) ([]float32, error)
}

func (m *MockEmbeddingProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	if m.EmbedFunc != nil {
		return m.EmbedFunc(ctx, text)
	}
	// 返回固定的向量
	return make([]float32, 1536), nil
}

// MockSemanticCache 模拟语义缓存
type MockSemanticCache struct {
	GetFunc func(ctx context.Context, queryVector []float32, threshold float32) (string, bool, error)
	SetFunc func(ctx context.Context, queryVector []float32, answer string) error
}

func (m *MockSemanticCache) Get(ctx context.Context, queryVector []float32, threshold float32) (string, bool, error) {
	if m.GetFunc != nil {
		return m.GetFunc(ctx, queryVector, threshold)
	}
	return "", false, nil
}

func (m *MockSemanticCache) Set(ctx context.Context, queryVector []float32, answer string) error {
	if m.SetFunc != nil {
		return m.SetFunc(ctx, queryVector, answer)
	}
	return nil
}
