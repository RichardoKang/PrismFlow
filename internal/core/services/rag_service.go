package services

import (
	"context"
	"fmt"
	"strings"
	"time"

	"PrismFlow/internal/core/domain"
	"PrismFlow/internal/core/ports"
	"PrismFlow/internal/infra/observability"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

// RAGService 负责编排整个检索生成流程
type RAGService struct {
	llm      ports.LLMProvider
	vectorDB ports.VectorStore
	embedder ports.EmbeddingProvider
	logger   *zap.Logger
}

// NewRAGService 构造函数 (依赖注入)
func NewRAGService(llm ports.LLMProvider, vdb ports.VectorStore, emb ports.EmbeddingProvider, logger *zap.Logger) *RAGService {
	return &RAGService{
		llm:      llm,
		vectorDB: vdb,
		embedder: emb,
		logger:   logger,
	}
}

// StreamChat 处理用户请求的主入口
// 返回一个只读 channel 用于流式传输，以及可能发生的错误 channel
func (s *RAGService) StreamChat(ctx context.Context, userQuery string) (<-chan string, <-chan error) {
	outChan := make(chan string)
	errChan := make(chan error, 1)

	go func() {
		defer close(outChan)
		defer close(errChan)

		// ========== 1. 创建 RAG 流程的根 Span ==========
		ctx, ragSpan := observability.StartSpan(ctx, "RAGService.StreamChat",
			trace.WithSpanKind(trace.SpanKindInternal),
		)
		defer ragSpan.End()

		// 记录查询信息
		observability.AddSpanAttributes(ragSpan,
			attribute.String("user.query", userQuery),
			attribute.Int("user.query_length", len(userQuery)),
		)

		s.logger.Info("Starting RAG flow", zap.String("query", userQuery))

		// ========== 2. Embedding 阶段 ==========
		embedCtx, embedSpan := observability.StartSpan(ctx, "Embedding",
			trace.WithSpanKind(trace.SpanKindClient),
		)
		embedSpan.SetAttributes(
			attribute.String("embedding.input", userQuery),
			attribute.String("embedding.provider", "mock"), // 可替换为真实 provider 名称
		)

		embedStart := time.Now()
		embedTimeoutCtx, embedCancel := context.WithTimeout(embedCtx, 3*time.Second)
		queryVector, err := s.embedder.Embed(embedTimeoutCtx, userQuery)
		embedCancel()
		embedDuration := time.Since(embedStart)

		embedSpan.SetAttributes(
			attribute.Float64("embedding.duration_ms", float64(embedDuration.Milliseconds())),
		)

		if err != nil {
			observability.SetSpanError(embedSpan, err)
			embedSpan.End()
			s.logger.Error("Embedding failed", zap.Error(err))
			errChan <- fmt.Errorf("failed to process query embedding: %w", err)
			return
		}

		embedSpan.SetAttributes(
			attribute.Int("embedding.vector_dim", len(queryVector)),
		)
		observability.SetSpanSuccess(embedSpan)
		observability.AddSpanEvent(embedSpan, "embedding_completed",
			attribute.Int("vector_dimensions", len(queryVector)),
		)
		embedSpan.End()

		// ========== 3. Vector Search 阶段 ==========
		searchCtx, searchSpan := observability.StartSpan(ctx, "VectorSearch",
			trace.WithSpanKind(trace.SpanKindClient),
		)
		topK := 5
		searchSpan.SetAttributes(
			attribute.Int("search.top_k", topK),
			attribute.String("search.provider", "mock_milvus"),
		)

		searchStart := time.Now()
		searchTimeoutCtx, searchCancel := context.WithTimeout(searchCtx, 5*time.Second)
		searchResults, err := s.vectorDB.Search(searchTimeoutCtx, queryVector, topK)
		searchCancel()
		searchDuration := time.Since(searchStart)

		searchSpan.SetAttributes(
			attribute.Float64("search.duration_ms", float64(searchDuration.Milliseconds())),
		)

		if err != nil {
			observability.SetSpanError(searchSpan, err)
			searchSpan.End()
			s.logger.Error("Vector search failed", zap.Error(err))
			errChan <- fmt.Errorf("knowledge retrieval failed: %w", err)
			return
		}

		// 记录搜索结果统计
		var maxScore, minScore float32 = 0, 1
		validDocsCount := 0
		for _, doc := range searchResults {
			if doc.Score > maxScore {
				maxScore = doc.Score
			}
			if doc.Score < minScore {
				minScore = doc.Score
			}
			if doc.Score >= 0.6 {
				validDocsCount++
			}
		}

		searchSpan.SetAttributes(
			attribute.Int("search.total_results", len(searchResults)),
			attribute.Int("search.valid_results", validDocsCount),
			attribute.Float64("search.max_score", float64(maxScore)),
			attribute.Float64("search.min_score", float64(minScore)),
		)
		observability.SetSpanSuccess(searchSpan)
		observability.AddSpanEvent(searchSpan, "search_completed",
			attribute.Int("docs_found", len(searchResults)),
			attribute.Int("docs_above_threshold", validDocsCount),
		)
		searchSpan.End()

		// ========== 4. Prompt Assembly 阶段 ==========
		_, promptSpan := observability.StartSpan(ctx, "PromptAssembly")
		promptStart := time.Now()

		systemPrompt := s.buildPrompt(userQuery, searchResults)

		promptDuration := time.Since(promptStart)
		promptSpan.SetAttributes(
			attribute.Int("prompt.system_length", len(systemPrompt)),
			attribute.Int("prompt.context_docs", validDocsCount),
			attribute.Float64("prompt.duration_ms", float64(promptDuration.Microseconds())/1000),
		)
		observability.SetSpanSuccess(promptSpan)
		promptSpan.End()

		// ========== 5. LLM Generation 阶段 ==========
		llmCtx, llmSpan := observability.StartSpan(ctx, "LLMGeneration",
			trace.WithSpanKind(trace.SpanKindClient),
		)
		llmSpan.SetAttributes(
			attribute.String("llm.provider", "deepseek"),
			attribute.String("llm.model", "deepseek-chat"),
			attribute.Int("llm.prompt_tokens_estimate", len(systemPrompt)/4), // 粗略估计
		)

		messages := []domain.Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userQuery},
		}

		llmStart := time.Now()
		tokenChan, llmErrChan := s.llm.ChatStream(llmCtx, messages)

		// 记录首 token 时间 (TTFT - Time To First Token)
		firstTokenReceived := false
		var ttft time.Duration
		tokenCount := 0
		var totalContent strings.Builder

		// ========== 6. Streaming 阶段 ==========
		_, streamSpan := observability.StartSpan(ctx, "StreamingResponse")

		for {
			select {
			case token, ok := <-tokenChan:
				if !ok {
					// 流结束，记录统计信息
					llmDuration := time.Since(llmStart)

					streamSpan.SetAttributes(
						attribute.Int("stream.token_count", tokenCount),
						attribute.Int("stream.total_chars", totalContent.Len()),
						attribute.Float64("stream.duration_ms", float64(llmDuration.Milliseconds())),
					)
					observability.SetSpanSuccess(streamSpan)
					streamSpan.End()

					llmSpan.SetAttributes(
						attribute.Int("llm.output_tokens", tokenCount),
						attribute.Float64("llm.total_duration_ms", float64(llmDuration.Milliseconds())),
						attribute.Float64("llm.ttft_ms", float64(ttft.Milliseconds())),
						attribute.Float64("llm.tokens_per_second", float64(tokenCount)/llmDuration.Seconds()),
					)
					observability.SetSpanSuccess(llmSpan)
					observability.AddSpanEvent(llmSpan, "generation_completed",
						attribute.Int("total_tokens", tokenCount),
						attribute.Float64("duration_seconds", llmDuration.Seconds()),
					)
					llmSpan.End()

					// 更新根 Span 统计
					ragSpan.SetAttributes(
						attribute.Float64("rag.total_duration_ms", float64(time.Since(llmStart).Milliseconds())+float64(embedDuration.Milliseconds())+float64(searchDuration.Milliseconds())),
						attribute.Int("rag.output_tokens", tokenCount),
						attribute.Bool("rag.success", true),
					)
					observability.SetSpanSuccess(ragSpan)
					return
				}

				// 记录首 token 时间
				if !firstTokenReceived {
					ttft = time.Since(llmStart)
					firstTokenReceived = true
					observability.AddSpanEvent(llmSpan, "first_token_received",
						attribute.Float64("ttft_ms", float64(ttft.Milliseconds())),
					)
				}

				tokenCount++
				totalContent.WriteString(token)

				select {
				case outChan <- token:
				case <-ctx.Done():
					streamSpan.SetAttributes(attribute.String("stream.cancel_reason", "client_disconnected"))
					streamSpan.End()
					llmSpan.SetAttributes(attribute.Bool("llm.cancelled", true))
					llmSpan.End()
					s.logger.Warn("Client disconnected during streaming")
					return
				}

			case err, ok := <-llmErrChan:
				if ok && err != nil {
					observability.SetSpanError(streamSpan, err)
					streamSpan.End()
					observability.SetSpanError(llmSpan, err)
					llmSpan.End()
					observability.SetSpanError(ragSpan, err)
					s.logger.Error("LLM stream error", zap.Error(err))
					errChan <- fmt.Errorf("LLM generation failed: %w", err)
					return
				}

			case <-ctx.Done():
				streamSpan.SetAttributes(attribute.String("stream.cancel_reason", "context_cancelled"))
				streamSpan.End()
				llmSpan.SetAttributes(attribute.Bool("llm.cancelled", true))
				llmSpan.End()
				ragSpan.SetAttributes(attribute.Bool("rag.cancelled", true))
				s.logger.Warn("Request context cancelled")
				return
			}
		}
	}()

	return outChan, errChan
}

// buildPrompt 构建提示词模板
func (s *RAGService) buildPrompt(_ string, docs []domain.SearchResult) string {
	var sb strings.Builder
	sb.WriteString("你是一个智能助手。请基于以下提供的上下文回答用户的问题。如果上下文没有相关信息，请诚实回答不知道后再给出你所认为的答案。\n\n")
	sb.WriteString("=== 上下文开始 ===\n")

	for i, doc := range docs {
		if doc.Score < 0.6 {
			continue
		}
		sb.WriteString(fmt.Sprintf("[%d] %s\n", i+1, doc.Content))
	}
	sb.WriteString("=== 上下文结束 ===\n")

	return sb.String()
}
