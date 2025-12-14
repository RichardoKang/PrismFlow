package observability

import (
	"context"
	"log"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"go.opentelemetry.io/otel/trace"
)

// Tracer 全局 Tracer 实例
var Tracer trace.Tracer

// InitTracer 初始化全局 Tracer
// 使用 OTLP HTTP Exporter 发送数据到 Jaeger
func InitTracer(serviceName string, otlpEndpoint string) func(context.Context) error {
	ctx := context.Background()

	// 1. 创建 OTLP HTTP Exporter (替代已废弃的 Jaeger Exporter)
	exporter, err := otlptracehttp.New(ctx,
		otlptracehttp.WithEndpoint(otlpEndpoint),
		otlptracehttp.WithInsecure(), // 本地开发不使用 TLS
	)
	if err != nil {
		log.Printf("Warning: Failed to create OTLP exporter: %v. Tracing disabled.", err)
		Tracer = otel.Tracer(serviceName)
		return func(ctx context.Context) error { return nil }
	}

	// 2. 创建 Resource (服务标识信息)
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(serviceName),
			semconv.ServiceVersion("1.0.0"),
			attribute.String("environment", "development"),
		),
	)
	if err != nil {
		log.Printf("Warning: Failed to create resource: %v", err)
		res = resource.Default()
	}

	// 3. 创建 TracerProvider
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.AlwaysSample()), // 开发环境全采样
	)

	// 4. 注册全局 TracerProvider
	otel.SetTracerProvider(tp)

	// 5. 设置全局 Propagator (用于跨服务传递 trace context)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	// 6. 初始化全局 Tracer
	Tracer = tp.Tracer(serviceName)

	log.Printf("Tracer initialized with OTLP: service=%s, endpoint=%s", serviceName, otlpEndpoint)

	return tp.Shutdown
}

// StartSpan 开始一个新的 Span
func StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	if Tracer == nil {
		Tracer = otel.Tracer("prismflow")
	}
	return Tracer.Start(ctx, name, opts...)
}

// SpanFromContext 从 context 获取当前 Span
func SpanFromContext(ctx context.Context) trace.Span {
	return trace.SpanFromContext(ctx)
}

// SetSpanError 在 Span 上记录错误
func SetSpanError(span trace.Span, err error) {
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
}

// SetSpanSuccess 标记 Span 成功
func SetSpanSuccess(span trace.Span) {
	span.SetStatus(codes.Ok, "success")
}

// AddSpanEvent 添加事件到 Span
func AddSpanEvent(span trace.Span, name string, attrs ...attribute.KeyValue) {
	span.AddEvent(name, trace.WithAttributes(attrs...))
}

// AddSpanAttributes 添加属性到 Span
func AddSpanAttributes(span trace.Span, attrs ...attribute.KeyValue) {
	span.SetAttributes(attrs...)
}
