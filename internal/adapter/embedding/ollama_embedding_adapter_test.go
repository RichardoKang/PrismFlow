package embedding

import (
	"context"
	"testing"
	"time"
)

func TestOllamaEmbed(t *testing.T) {
	adapter := NewOllamaEmbeddingAdapter(OllamaConfig{
		BaseURL: "http://localhost:11434",
		Model:   "nomic-embed-text",
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	vector, err := adapter.Embed(ctx, "测试中文文本")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	if len(vector) == 0 {
		t.Fatal("Embed returned empty vector")
	}

	t.Logf("Success! Vector dimension: %d", len(vector))
	t.Logf("First 5 values: %v", vector[:5])
}
