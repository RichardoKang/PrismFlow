package domain

// Message 代表 LLM 的对话消息
type Message struct {
	Role    string // "system", "user", "assistant"
	Content string
}

// SearchResult 代表通用的检索结果
type SearchResult struct {
	ID      string
	Content string
	Score   float32
	Meta    map[string]interface{}
}
