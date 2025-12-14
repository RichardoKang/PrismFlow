package llm

import (
	"context"
	"errors"
	"io"

	"PrismFlow/internal/core/domain"
	"PrismFlow/internal/core/ports"

	"github.com/sashabaranov/go-openai" // 官方社区推荐的 client
)

type OpenAIAdapter struct {
	client *openai.Client
	model  string
}

// 确保 OpenAIAdapter 实现了 ports.LLMProvider 接口
// 这是一个编译期检查技巧
var _ ports.LLMProvider = (*OpenAIAdapter)(nil)

func NewOpenAIAdapter(apiKey string, model string) *OpenAIAdapter {
	return &OpenAIAdapter{
		client: openai.NewClient(apiKey),
		model:  model,
	}
}

func NewDeepSeekAdapter(apiKey string, model string) *OpenAIAdapter {
	// 1. 使用 OpenAI 的默认配置作为基础
	config := openai.DefaultConfig(apiKey)

	// 2. 关键修改：将 BaseURL 指向 DeepSeek 的官方地址
	config.BaseURL = "https://api.deepseek.com"

	// 3. 创建带有自定义配置的 Client
	return &OpenAIAdapter{
		client: openai.NewClientWithConfig(config),
		model:  model,
	}
}

// ChatStream 实现流式调用
func (a *OpenAIAdapter) ChatStream(ctx context.Context, messages []domain.Message) (<-chan string, <-chan error) {
	// 1. 转换 domain.Message 到 openai.ChatCompletionMessage
	reqMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, m := range messages {
		reqMessages[i] = openai.ChatCompletionMessage{
			Role:    m.Role,
			Content: m.Content,
		}
	}

	req := openai.ChatCompletionRequest{
		Model:    a.model,
		Messages: reqMessages,
		Stream:   true, // 开启流式
	}

	// 2. 创建 output channels
	tokenChan := make(chan string)
	errChan := make(chan error, 1) // buffer 1 防止阻塞

	// 3. 启动 Goroutine 处理流
	go func() {
		defer close(tokenChan)
		defer close(errChan)

		stream, err := a.client.CreateChatCompletionStream(ctx, req)
		if err != nil {
			errChan <- err
			return
		}
		defer stream.Close()

		for {
			response, err := stream.Recv()

			// 处理流结束
			if errors.Is(err, io.EOF) {
				return
			}

			// 处理错误
			if err != nil {
				errChan <- err
				return
			}

			// 提取内容并通过 channel 发送
			if len(response.Choices) > 0 {
				content := response.Choices[0].Delta.Content
				if content != "" {
					// 检查 ctx 是否取消 (比如客户端断开)
					select {
					case tokenChan <- content:
					case <-ctx.Done():
						return // 停止处理
					}
				}
			}
		}
	}()

	return tokenChan, errChan
}

// Chat 方法实现略...
func (a *OpenAIAdapter) Chat(ctx context.Context, messages []domain.Message) (string, error) {
	// TODO
	return "", nil
}
