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
	llm            ports.LLMProvider
	vectorDB       ports.VectorStore
	retriever      ports.HybridRetriever // 混合检索器，为 nil 时降级为纯向量检索
	embedder       ports.EmbeddingProvider
	reranker       ports.RerankerProvider
	logger         *zap.Logger
	scoreThreshold float32
	topK           int
	rerankTopN     int
}

// NewRAGService 构造函数 (依赖注入)
func NewRAGService(llm ports.LLMProvider, vdb ports.VectorStore, emb ports.EmbeddingProvider, reranker ports.RerankerProvider, retriever ports.HybridRetriever, logger *zap.Logger, scoreThreshold float32, topK int, rerankTopN int) *RAGService {
	return &RAGService{
		llm:            llm,
		vectorDB:       vdb,
		retriever:      retriever,
		embedder:       emb,
		reranker:       reranker,
		logger:         logger,
		scoreThreshold: scoreThreshold,
		topK:           topK,
		rerankTopN:     rerankTopN,
	}
}

// StreamChat 处理用户请求的主入口
// queryVector 为可选参数，如果缓存中间件已计算则复用，避免双重 embedding
func (s *RAGService) StreamChat(ctx context.Context, userQuery string, queryVector []float32) (<-chan string, <-chan error) {
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

		// ========== 2. Embedding 阶段（复用已有向量或重新计算） ==========
		if len(queryVector) > 0 {
			s.logger.Info("Reusing query vector from cache middleware", zap.Int("dim", len(queryVector)))
			observability.AddSpanEvent(ragSpan, "embedding_reused",
				attribute.Int("vector_dimensions", len(queryVector)),
			)
		} else {
			embedCtx, embedSpan := observability.StartSpan(ctx, "Embedding",
				trace.WithSpanKind(trace.SpanKindClient),
			)
			embedSpan.SetAttributes(
				attribute.String("embedding.input", userQuery),
				attribute.String("embedding.provider", "ollama"),
			)

			embedStart := time.Now()
			embedTimeoutCtx, embedCancel := context.WithTimeout(embedCtx, 3*time.Second)
			var err error
			queryVector, err = s.embedder.Embed(embedTimeoutCtx, userQuery)
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
		}

		// ========== 3. Retrieval 阶段（Hybrid Search 或纯向量检索） ==========
		searchCtx, searchSpan := observability.StartSpan(ctx, "Retrieval",
			trace.WithSpanKind(trace.SpanKindClient),
		)
		topK := s.topK
		searchSpan.SetAttributes(
			attribute.Int("search.top_k", topK),
			attribute.Bool("search.hybrid_enabled", s.retriever != nil),
		)

		searchStart := time.Now()
		searchTimeoutCtx, searchCancel := context.WithTimeout(searchCtx, 5*time.Second)

		var searchResults []domain.SearchResult
		var searchErr error
		if s.retriever != nil {
			searchResults, searchErr = s.retriever.Search(searchTimeoutCtx, userQuery, queryVector, topK)
			searchSpan.SetAttributes(attribute.String("search.mode", "hybrid"))
		} else {
			searchResults, searchErr = s.vectorDB.Search(searchTimeoutCtx, queryVector, topK)
			searchSpan.SetAttributes(attribute.String("search.mode", "vector_only"))
		}
		searchCancel()
		searchDuration := time.Since(searchStart)

		searchSpan.SetAttributes(
			attribute.Float64("search.duration_ms", float64(searchDuration.Milliseconds())),
		)

		if searchErr != nil {
			observability.SetSpanError(searchSpan, searchErr)
			searchSpan.End()
			s.logger.Error("Retrieval failed", zap.Error(searchErr))
			errChan <- fmt.Errorf("knowledge retrieval failed: %w", searchErr)
			return
		}

		// 记录搜索结果统计
		var maxScore, minScore float32 = 0, 1
		validDocsCount := 0
		for i, doc := range searchResults {
			s.logger.Info("Search result",
				zap.Int("rank", i+1),
				zap.Float32("score", doc.Score),
				zap.Int("content_len", len(doc.Content)),
				zap.String("content_preview", truncate(doc.Content, 80)),
			)
			if doc.Score > maxScore {
				maxScore = doc.Score
			}
			if doc.Score < minScore {
				minScore = doc.Score
			}
			if doc.Score >= s.scoreThreshold {
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

		// ========== 4. Rerank 阶段 ==========
		if s.reranker != nil && len(searchResults) > 0 {
			_, rerankSpan := observability.StartSpan(ctx, "Rerank")
			rerankSpan.SetAttributes(
				attribute.Int("rerank.input_docs", len(searchResults)),
				attribute.Int("rerank.top_n", s.rerankTopN),
			)

			rerankStart := time.Now()
			reranked, rerankErr := s.reranker.Rerank(ctx, userQuery, searchResults, s.rerankTopN)
			rerankDuration := time.Since(rerankStart)

			rerankSpan.SetAttributes(
				attribute.Float64("rerank.duration_ms", float64(rerankDuration.Milliseconds())),
			)

			if rerankErr != nil {
				observability.SetSpanError(rerankSpan, rerankErr)
				s.logger.Warn("Rerank failed, using original results", zap.Error(rerankErr))
			} else {
				searchResults = reranked
				rerankSpan.SetAttributes(attribute.Int("rerank.output_docs", len(reranked)))
				observability.SetSpanSuccess(rerankSpan)
			}
			rerankSpan.End()
		}

		// ========== 5. Prompt Assembly 阶段 ==========
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
						attribute.Float64("rag.total_duration_ms", float64(time.Since(llmStart).Milliseconds())+float64(searchDuration.Milliseconds())),
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

	idx := 0
	for _, doc := range docs {
		if doc.Score < s.scoreThreshold {
			continue
		}
		idx++
		sb.WriteString(fmt.Sprintf("[%d] %s\n", idx, doc.Content))
	}
	sb.WriteString("=== 上下文结束 ===\n")

	return sb.String()
}

// truncate 截断字符串用于日志输出
func truncate(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "..."
}
