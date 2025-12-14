package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"PrismFlow/internal/core/ports"
	"PrismFlow/internal/infra/observability"

	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/otel/attribute"
	"go.uber.org/zap"
)

// chatRequest 用于解析请求体中的 query
type chatRequest struct {
	Query string `json:"query"`
}

// SemanticCacheMiddleware 构建缓存中间件
func SemanticCacheMiddleware(
	embedder ports.EmbeddingProvider,
	cache ports.SemanticCache,
	logger *zap.Logger,
) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 创建缓存检查的 Span
		ctx, cacheSpan := observability.StartSpan(c.Request.Context(), "SemanticCache.Check")
		c.Request = c.Request.WithContext(ctx)

		startTime := time.Now()

		// 1. 读取并复制 Request Body
		bodyBytes, err := io.ReadAll(c.Request.Body)
		if err != nil {
			cacheSpan.SetAttributes(attribute.String("cache.skip_reason", "body_read_error"))
			cacheSpan.End()
			logger.Warn("Failed to read request body", zap.Error(err))
			c.Next()
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

		// 2. 解析 Query
		var req chatRequest
		if err := json.Unmarshal(bodyBytes, &req); err != nil || req.Query == "" {
			cacheSpan.SetAttributes(attribute.String("cache.skip_reason", "invalid_query"))
			cacheSpan.End()
			c.Next()
			return
		}

		userQuery := req.Query
		cacheSpan.SetAttributes(
			attribute.String("cache.query", userQuery),
			attribute.Int("cache.query_length", len(userQuery)),
		)

		// 3. 计算向量（用于缓存查找）
		_, embedSpan := observability.StartSpan(ctx, "SemanticCache.Embedding")
		embCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)

		embedStart := time.Now()
		vector, err := embedder.Embed(embCtx, userQuery)
		cancel()
		embedDuration := time.Since(embedStart)

		embedSpan.SetAttributes(
			attribute.Float64("embedding.duration_ms", float64(embedDuration.Milliseconds())),
		)

		if err != nil {
			observability.SetSpanError(embedSpan, err)
			embedSpan.End()
			cacheSpan.SetAttributes(attribute.String("cache.skip_reason", "embedding_error"))
			cacheSpan.End()
			logger.Warn("Cache embedding failed", zap.Error(err))
			c.Next()
			return
		}
		embedSpan.SetAttributes(attribute.Int("embedding.vector_dim", len(vector)))
		observability.SetSpanSuccess(embedSpan)
		embedSpan.End()

		// 4. 查缓存
		_, lookupSpan := observability.StartSpan(ctx, "SemanticCache.Lookup")
		lookupStart := time.Now()
		cachedAnswer, hit, err := cache.Get(embCtx, vector, 0.95)
		lookupDuration := time.Since(lookupStart)

		lookupSpan.SetAttributes(
			attribute.Float64("lookup.duration_ms", float64(lookupDuration.Milliseconds())),
			attribute.Float64("lookup.threshold", 0.95),
		)

		if err == nil && hit {
			// === 缓存命中 ===
			lookupSpan.SetAttributes(
				attribute.Bool("lookup.hit", true),
				attribute.Int("lookup.cached_length", len(cachedAnswer)),
			)
			observability.SetSpanSuccess(lookupSpan)
			lookupSpan.End()

			cacheSpan.SetAttributes(
				attribute.Bool("cache.hit", true),
				attribute.Float64("cache.total_duration_ms", float64(time.Since(startTime).Milliseconds())),
			)
			observability.AddSpanEvent(cacheSpan, "cache_hit",
				attribute.Int("cached_response_length", len(cachedAnswer)),
			)
			observability.SetSpanSuccess(cacheSpan)
			cacheSpan.End()

			logger.Info("Semantic Cache Hit!", zap.String("query", userQuery))
			streamCachedResponse(c, cachedAnswer)
			c.Abort()
			return
		}

		// === 缓存未命中 ===
		lookupSpan.SetAttributes(attribute.Bool("lookup.hit", false))
		if err != nil {
			lookupSpan.SetAttributes(attribute.String("lookup.error", err.Error()))
		}
		lookupSpan.End()

		cacheSpan.SetAttributes(
			attribute.Bool("cache.hit", false),
			attribute.Float64("cache.lookup_duration_ms", float64(time.Since(startTime).Milliseconds())),
		)
		observability.AddSpanEvent(cacheSpan, "cache_miss")
		cacheSpan.End()

		// === MISS: 准备捕获响应 ===
		recorder := &streamRecorder{
			ResponseWriter: c.Writer,
			body:           &bytes.Buffer{},
		}
		c.Writer = recorder

		// 5. 继续执行 RAG 流程
		c.Next()

		// 6. 异步写入缓存
		fullResponse := extractContentFromSSE(recorder.body.String())
		if c.Writer.Status() == 200 && len(fullResponse) > 0 {
			go func(vec []float32, ans string) {
				writeCtx, writeSpan := observability.StartSpan(context.Background(), "SemanticCache.Write")
				defer writeSpan.End()

				writeCtx, writeCancel := context.WithTimeout(writeCtx, 5*time.Second)
				defer writeCancel()

				writeStart := time.Now()
				if err := cache.Set(writeCtx, vec, ans); err != nil {
					writeSpan.SetAttributes(attribute.String("write.error", err.Error()))
					observability.SetSpanError(writeSpan, err)
					logger.Error("Failed to update semantic cache", zap.Error(err))
				} else {
					writeSpan.SetAttributes(
						attribute.Int("write.content_length", len(ans)),
						attribute.Float64("write.duration_ms", float64(time.Since(writeStart).Milliseconds())),
					)
					observability.SetSpanSuccess(writeSpan)
					logger.Info("Semantic cache updated", zap.Int("content_length", len(ans)))
				}
			}(vector, fullResponse)
		}
	}
}

// extractContentFromSSE 从 SSE 格式的响应中提取纯文本内容
func extractContentFromSSE(sseData string) string {
	var sb strings.Builder
	lines := strings.Split(sseData, "\n")

	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			// 尝试解析 JSON 格式
			var msg struct {
				Content string `json:"content"`
				Done    bool   `json:"done"`
			}
			if err := json.Unmarshal([]byte(data), &msg); err == nil {
				if !msg.Done && msg.Content != "" {
					sb.WriteString(msg.Content)
				}
			}
		}
	}

	return sb.String()
}

// streamRecorder 劫持 ResponseWriter，同时写入客户端和 Buffer
type streamRecorder struct {
	gin.ResponseWriter
	body *bytes.Buffer
}

// Write 重写 Write 方法，实现"双写"
func (w *streamRecorder) Write(b []byte) (int, error) {
	w.body.Write(b)
	return w.ResponseWriter.Write(b)
}

// streamCachedResponse 将完整文本伪装成 SSE 流发送
func streamCachedResponse(c *gin.Context, text string) {
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("X-Accel-Buffering", "no")

	// 使用 JSON 格式，与 Handler 保持一致
	type SSEMessage struct {
		Content string `json:"content,omitempty"`
		Done    bool   `json:"done,omitempty"`
	}

	// 模拟打字机效果
	chunkSize := 10
	runes := []rune(text)

	for i := 0; i < len(runes); i += chunkSize {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}

		chunk := string(runes[i:end])
		data, _ := json.Marshal(SSEMessage{Content: chunk})
		_, _ = fmt.Fprintf(c.Writer, "data: %s\n\n", data)
		c.Writer.Flush()

		time.Sleep(10 * time.Millisecond)
	}

	// 发送结束标记
	data, _ := json.Marshal(SSEMessage{Done: true})
	_, _ = fmt.Fprintf(c.Writer, "data: %s\n\n", data)
	c.Writer.Flush()
}
