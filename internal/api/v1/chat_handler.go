package v1

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"PrismFlow/internal/core/services"
	"PrismFlow/internal/infra/observability"

	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
)

// ChatRequest 请求结构体
type ChatRequest struct {
	Query string `json:"query" binding:"required"`
}

// SSEMessage SSE 消消息格式
type SSEMessage struct {
	Content string `json:"content,omitempty"`
	Error   string `json:"error,omitempty"`
	Done    bool   `json:"done,omitempty"`
}

// ChatHandler 定义 HTTP 处理层依赖
type ChatHandler struct {
	ragService *services.RAGService
	logger     *zap.Logger
}

// NewChatHandler 构造函数
func NewChatHandler(ragService *services.RAGService, logger *zap.Logger) *ChatHandler {
	return &ChatHandler{
		ragService: ragService,
		logger:     logger,
	}
}

// HandleChat 处理对话请求
func (h *ChatHandler) HandleChat(c *gin.Context) {
	startTime := time.Now()

	// 创建 Handler 层的 Span
	ctx, handlerSpan := observability.StartSpan(c.Request.Context(), "ChatHandler.HandleChat",
		trace.WithSpanKind(trace.SpanKindServer),
	)
	defer handlerSpan.End()

	// 记录请求信息
	handlerSpan.SetAttributes(
		attribute.String("http.method", c.Request.Method),
		attribute.String("http.url", c.Request.URL.Path),
		attribute.String("http.client_ip", c.ClientIP()),
	)

	// 1. 解析请求
	_, parseSpan := observability.StartSpan(ctx, "ParseRequest")
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		observability.SetSpanError(parseSpan, err)
		parseSpan.End()
		handlerSpan.SetAttributes(
			attribute.Int("http.status_code", http.StatusBadRequest),
			attribute.String("error.type", "invalid_request"),
		)
		h.logger.Warn("Invalid request", zap.Error(err))
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	parseSpan.SetAttributes(
		attribute.String("request.query", req.Query),
		attribute.Int("request.query_length", len(req.Query)),
	)
	observability.SetSpanSuccess(parseSpan)
	parseSpan.End()

	// 更新 handler span 的查询信息
	handlerSpan.SetAttributes(
		attribute.String("chat.query", req.Query),
	)

	// 2. 将 query 存入 context
	c.Set("user_query", req.Query)

	// 3. 检查是否支持 Flush
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		handlerSpan.SetAttributes(
			attribute.Int("http.status_code", http.StatusInternalServerError),
			attribute.String("error.type", "streaming_not_supported"),
		)
		h.logger.Error("Streaming not supported")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "streaming not supported"})
		return
	}

	// 4. 设置 SSE Headers
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")

	// 5. 调用编排服务（传递带 trace 的 context）
	tokenChan, errChan := h.ragService.StreamChat(ctx, req.Query)

	// 6. 创建 SSE 发送的 Span
	_, sseSpan := observability.StartSpan(ctx, "SSE.SendResponse")
	sseSpan.SetAttributes(
		attribute.String("sse.content_type", "text/event-stream"),
	)

	headerFlushed := false
	tokenCount := 0
	firstTokenTime := time.Time{}

	// 7. 推送流数据
	for {
		select {
		case token, ok := <-tokenChan:
			if !ok {
				// 流正常结束
				h.sendSSEMessage(c.Writer, SSEMessage{Done: true})
				flusher.Flush()

				totalDuration := time.Since(startTime)
				sseSpan.SetAttributes(
					attribute.Int("sse.token_count", tokenCount),
					attribute.Float64("sse.duration_ms", float64(totalDuration.Milliseconds())),
					attribute.Bool("sse.completed", true),
				)
				if !firstTokenTime.IsZero() {
					sseSpan.SetAttributes(
						attribute.Float64("sse.ttft_ms", float64(firstTokenTime.Sub(startTime).Milliseconds())),
					)
				}
				observability.SetSpanSuccess(sseSpan)
				sseSpan.End()

				handlerSpan.SetAttributes(
					attribute.Int("http.status_code", http.StatusOK),
					attribute.Int("chat.output_tokens", tokenCount),
					attribute.Float64("chat.total_duration_ms", float64(totalDuration.Milliseconds())),
				)
				observability.SetSpanSuccess(handlerSpan)
				return
			}

			if !headerFlushed {
				flusher.Flush()
				headerFlushed = true
				firstTokenTime = time.Now()
				observability.AddSpanEvent(sseSpan, "first_token_sent",
					attribute.Float64("ttft_ms", float64(time.Since(startTime).Milliseconds())),
				)
			}

			tokenCount++
			h.sendSSEMessage(c.Writer, SSEMessage{Content: token})
			flusher.Flush()

		case err, ok := <-errChan:
			if ok && err != nil {
				h.logger.Error("Stream error", zap.Error(err))
				sseSpan.SetAttributes(attribute.String("sse.error", err.Error()))
				observability.SetSpanError(sseSpan, err)
				sseSpan.End()

				handlerSpan.SetAttributes(
					attribute.Int("http.status_code", http.StatusInternalServerError),
					attribute.String("error.message", err.Error()),
				)
				observability.SetSpanError(handlerSpan, err)

				if headerFlushed {
					h.sendSSEEvent(c.Writer, "error", SSEMessage{Error: err.Error()})
				} else {
					c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				}
				flusher.Flush()
				return
			}

		case <-c.Request.Context().Done():
			sseSpan.SetAttributes(
				attribute.String("sse.cancel_reason", "client_disconnected"),
				attribute.Int("sse.tokens_sent", tokenCount),
			)
			sseSpan.End()

			handlerSpan.SetAttributes(
				attribute.Bool("http.client_disconnected", true),
			)
			h.logger.Info("Client disconnected")
			return
		}
	}
}

// sendSSEMessage 发送标准 SSE 数据消息
func (h *ChatHandler) sendSSEMessage(w http.ResponseWriter, msg SSEMessage) {
	data, _ := json.Marshal(msg)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
}

// sendSSEEvent 发送带事件类型的 SSE 消息
func (h *ChatHandler) sendSSEEvent(w http.ResponseWriter, event string, msg SSEMessage) {
	data, _ := json.Marshal(msg)
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
}
