package config

import (
	"fmt"
	"os"

	"github.com/goccy/go-yaml"
)

// Config 应用配置根结构
type Config struct {
	Server    ServerConfig    `yaml:"server"`
	LLM       LLMConfig       `yaml:"llm"`
	Embedding EmbeddingConfig `yaml:"embedding"`
	VectorDB  VectorDBConfig  `yaml:"vector_db"`
	Redis     RedisConfig     `yaml:"redis"`
	Trace     TraceConfig     `yaml:"trace"`
}

type ServerConfig struct {
	Port int    `yaml:"port"`
	Name string `yaml:"name"`
}

type LLMConfig struct {
	Provider string `yaml:"provider"`
	APIKey   string `yaml:"api_key"`
	Model    string `yaml:"model"`
	BaseURL  string `yaml:"base_url"`
}

type EmbeddingConfig struct {
	Provider string `yaml:"provider"` // ollama / openai
	APIKey   string `yaml:"api_key"`
	BaseURL  string `yaml:"base_url"`
	Model    string `yaml:"model"`
}

type VectorDBConfig struct {
	Host           string `yaml:"host"`
	Port           int    `yaml:"port"`
	CollectionName string `yaml:"collection_name"`
	Dimension      int    `yaml:"dimension"`
}

type RedisConfig struct {
	Addr     string `yaml:"addr"`
	Password string `yaml:"password"`
}

type TraceConfig struct {
	Endpoint string `yaml:"endpoint"`
}

// Load 从文件加载配置，支持环境变量覆盖
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// 环境变量覆盖（优先级高于配置文件）
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		cfg.LLM.APIKey = apiKey
	}
	if apiKey := os.Getenv("EMBEDDING_API_KEY"); apiKey != "" {
		cfg.Embedding.APIKey = apiKey
	}
	if redisAddr := os.Getenv("REDIS_ADDR"); redisAddr != "" {
		cfg.Redis.Addr = redisAddr
	}
	if redisPassword := os.Getenv("REDIS_PASSWORD"); redisPassword != "" {
		cfg.Redis.Password = redisPassword
	}

	return &cfg, nil
}

// VectorDBAddr 返回向量数据库地址
func (c *Config) VectorDBAddr() string {
	return fmt.Sprintf("%s:%d", c.VectorDB.Host, c.VectorDB.Port)
}
