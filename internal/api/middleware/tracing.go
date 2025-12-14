package middleware

import (
	"time"

	"PrismFlow/internal/infra/observability"

	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

func TracingMiddleware(serviceName string) gin.HandlerFunc {
	return func(c *gin.Context) {
		startTime := time.Now()

		// 1. 从 HTTP Header 中提取上游传来的 TraceID (如果是微服务调用)
		ctx := otel.GetTextMapPropagator().Extract(c.Request.Context(), propagation.HeaderCarrier(c.Request.Header))

		// 2. 创建根 Span
		ctx, span := observability.StartSpan(ctx, "HTTP "+c.Request.Method+" "+c.Request.URL.Path,
			trace.WithSpanKind(trace.SpanKindServer),
		)
		defer span.End()

		// 3. 注入请求元信息
		span.SetAttributes(
			attribute.String("http.method", c.Request.Method),
			attribute.String("http.url", c.Request.URL.String()),
			attribute.String("http.path", c.Request.URL.Path),
			attribute.String("http.host", c.Request.Host),
			attribute.String("http.user_agent", c.Request.UserAgent()),
			attribute.String("http.client_ip", c.ClientIP()),
			attribute.String("http.scheme", c.Request.URL.Scheme),
		)

		// 4. 如果有 Content-Length，记录请求大小
		if c.Request.ContentLength > 0 {
			span.SetAttributes(attribute.Int64("http.request_content_length", c.Request.ContentLength))
		}

		// 5. 将带有 Trace 的 Context 塞回 Request
		c.Request = c.Request.WithContext(ctx)

		// 6. 执行后续处理
		c.Next()

		// 7. 记录响应信息
		duration := time.Since(startTime)
		statusCode := c.Writer.Status()

		span.SetAttributes(
			attribute.Int("http.status_code", statusCode),
			attribute.Int("http.response_size", c.Writer.Size()),
			attribute.Float64("http.duration_ms", float64(duration.Milliseconds())),
		)

		// 8. 根据状态码设置 Span 状态
		if statusCode >= 400 {
			span.SetAttributes(attribute.Bool("error", true))
			if statusCode >= 500 {
				span.SetAttributes(attribute.String("error.type", "server_error"))
			} else {
				span.SetAttributes(attribute.String("error.type", "client_error"))
			}
		} else {
			observability.SetSpanSuccess(span)
		}

		// 9. 添加完成事件
		observability.AddSpanEvent(span, "request_completed",
			attribute.Int("status_code", statusCode),
			attribute.Float64("duration_ms", float64(duration.Milliseconds())),
		)
	}
}
