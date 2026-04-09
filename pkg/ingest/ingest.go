package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"

	"PrismFlow/internal/adapter/embedding"
	"PrismFlow/internal/adapter/vectordb"
	"PrismFlow/internal/config"
	"PrismFlow/internal/core/ports"
)

// Document 表示分割后的文档块
type Document struct {
	Title   string // 章节标题
	Content string // 内容
	Source  string // 来源文件
}

func main() {
	// 命令行参数
	configPath := flag.String("config", "configs/config.yaml", "配置文件路径")
	mdPath := flag.String("file", "", "要导入的 Markdown 文件路径")
	chunkSize := flag.Int("chunk-size", 500, "每个文档块的最大字符数")
	overlap := flag.Int("overlap", 50, "块之间的重叠字符数")
	batchSize := flag.Int("batch-size", 20, "批量处理大小")
	flag.Parse()

	if *mdPath == "" {
		log.Fatal("请指定 Markdown 文件路径: -file <path>")
	}

	// 加载配置
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("加载配置失败: %v", err)
	}

	// 初始化 Milvus
	log.Printf("正在连接 Milvus: %s", cfg.VectorDBAddr())
	milvusCfg := vectordb.MilvusConfig{
		Address:        cfg.VectorDBAddr(),
		CollectionName: cfg.VectorDB.CollectionName,
		Dimension:      cfg.VectorDB.Dimension,
	}
	milvusAdapter, err := vectordb.NewMilvusAdapter(milvusCfg)
	if err != nil {
		log.Fatalf("连接 Milvus 失败: %v", err)
	}
	defer milvusAdapter.Close()
	log.Println("Milvus 连接成功")

	// 初始化 Embedding Adapter
	var embedAdapter ports.EmbeddingProvider
	switch cfg.Embedding.Provider {
	case "ollama":
		embedAdapter = embedding.NewOllamaEmbeddingAdapter(embedding.OllamaConfig{
			BaseURL: cfg.Embedding.BaseURL,
			Model:   cfg.Embedding.Model,
		})
		log.Printf("使用 Ollama embedding: %s", cfg.Embedding.Model)
	case "llamacpp":
		embedAdapter = embedding.NewLlamaCppEmbeddingAdapter(cfg.Embedding.BaseURL)
		log.Printf("使用 llama.cpp embedding: %s", cfg.Embedding.BaseURL)
	default:
		embedAdapter = embedding.NewMockEmbeddingAdapter()
		log.Printf("使用 Mock embedding（维度: %d）", cfg.VectorDB.Dimension)
	}

	// 需要 EmbedBatch 支持
	batchEmbedder, hasBatch := embedAdapter.(interface {
		EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
	})
	if !hasBatch {
		log.Fatal("Embedding adapter 不支持 EmbedBatch")
	}

	// 读取并分割 Markdown 文件
	log.Printf("正在读取文件: %s", *mdPath)
	docs, err := splitMarkdown(*mdPath, *chunkSize, *overlap)
	if err != nil {
		log.Fatalf("分割文档失败: %v", err)
	}
	log.Printf("共分割为 %d 个文档块", len(docs))

	// 批量导入到 Milvus
	ctx := context.Background()
	successCount := 0
	failCount := 0

	for i := 0; i < len(docs); i += *batchSize {
		end := i + *batchSize
		if end > len(docs) {
			end = len(docs)
		}
		batch := docs[i:end]

		// 收集当前批次的文本
		texts := make([]string, len(batch))
		for j, doc := range batch {
			texts[j] = doc.Content
		}

		// 批量生成 embedding
		vectors, err := batchEmbedder.EmbedBatch(ctx, texts)
		if err != nil {
			log.Printf("[batch %d-%d] 批量 embedding 失败: %v", i+1, end, err)
			failCount += len(batch)
			continue
		}

		// 构建批量存储参数
		contents := make([]string, len(batch))
		metas := make([]map[string]interface{}, len(batch))
		for j, doc := range batch {
			contents[j] = doc.Content
			metas[j] = map[string]interface{}{
				"title":  doc.Title,
				"source": doc.Source,
			}
		}

		// 批量写入 Milvus（统一 Flush）
		if err := milvusAdapter.StoreBatch(ctx, vectors, contents, metas); err != nil {
			log.Printf("[batch %d-%d] 批量写入 Milvus 失败: %v", i+1, end, err)
			failCount += len(batch)
			continue
		}

		successCount += len(batch)
		log.Printf("进度: %d/%d (成功: %d, 失败: %d)", end, len(docs), successCount, failCount)
	}

	log.Printf("导入完成! 成功: %d, 失败: %d, 总计: %d", successCount, failCount, len(docs))
}

// Markdown 格式清理正则
var (
	titleRegex    = regexp.MustCompile(`^#+\s+(.+)$`)
	codeBlockRe   = regexp.MustCompile("(?s)```.*?```")
	linkRe        = regexp.MustCompile(`\[([^\]]+)\]\([^)]+\)`)
	boldItalicRe  = regexp.MustCompile(`[*_]{1,3}([^*_]+)[*_]{1,3}`)
	imageRe       = regexp.MustCompile(`!\[[^\]]*\]\([^)]+\)`)
	htmlTagRe     = regexp.MustCompile(`<[^>]+>`)
	inlineCodeRe  = regexp.MustCompile("`([^`]+)`")
)

// stripMarkdown 去除 Markdown 格式符号
func stripMarkdown(text string) string {
	result := text
	result = codeBlockRe.ReplaceAllString(result, "")
	result = imageRe.ReplaceAllString(result, "")
	result = linkRe.ReplaceAllString(result, "$1")
	result = boldItalicRe.ReplaceAllString(result, "$1")
	result = htmlTagRe.ReplaceAllString(result, "")
	result = inlineCodeRe.ReplaceAllString(result, "$1")
	return result
}

// splitMarkdown 按章节和段落分割 Markdown 文件
func splitMarkdown(filePath string, chunkSize, overlap int) ([]Document, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("打开文件失败: %w", err)
	}
	defer file.Close()

	var docs []Document
	var currentTitle string
	var currentContent strings.Builder

	scanner := bufio.NewScanner(file)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	fileName := filePath[strings.LastIndex(filePath, "/")+1:]

	for scanner.Scan() {
		line := scanner.Text()

		// 检测标题行
		if matches := titleRegex.FindStringSubmatch(line); len(matches) > 1 {
			// 保存之前的内容
			if currentContent.Len() > 0 {
				cleaned := stripMarkdown(currentContent.String())
				chunks := splitIntoChunks(cleaned, chunkSize, overlap)
				for _, chunk := range chunks {
					if strings.TrimSpace(chunk) != "" {
						docs = append(docs, Document{
							Title:   currentTitle,
							Content: fmt.Sprintf("【%s】\n%s", currentTitle, chunk),
							Source:  fileName,
						})
					}
				}
			}
			currentTitle = matches[1]
			currentContent.Reset()
		} else {
			// 跳过图片行
			if strings.HasPrefix(strings.TrimSpace(line), "![") {
				continue
			}
			currentContent.WriteString(line)
			currentContent.WriteString("\n")
		}
	}

	// 处理最后一个章节
	if currentContent.Len() > 0 {
		cleaned := stripMarkdown(currentContent.String())
		chunks := splitIntoChunks(cleaned, chunkSize, overlap)
		for _, chunk := range chunks {
			if strings.TrimSpace(chunk) != "" {
				docs = append(docs, Document{
					Title:   currentTitle,
					Content: fmt.Sprintf("【%s】\n%s", currentTitle, chunk),
					Source:  fileName,
				})
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("读取文件失败: %w", err)
	}

	return docs, nil
}

// splitIntoChunks 优先按段落/句子边界分割，带重叠
func splitIntoChunks(text string, chunkSize, overlap int) []string {
	text = strings.TrimSpace(text)
	if len(text) == 0 {
		return nil
	}

	runes := []rune(text)
	if len(runes) <= chunkSize {
		return []string{text}
	}

	var chunks []string
	start := 0

	for start < len(runes) {
		end := start + chunkSize
		if end > len(runes) {
			end = len(runes)
		}

		// 优先在段落边界断开
		if end < len(runes) {
			bestBreak := -1
			// 先找段落边界（双换行）
			for i := end; i > start+chunkSize/2; i-- {
				if i+1 < len(runes) && runes[i] == '\n' && runes[i-1] == '\n' {
					bestBreak = i + 1
					break
				}
			}
			// 再找句子边界
			if bestBreak == -1 {
				for i := end; i > start+chunkSize/2; i-- {
					if runes[i] == '。' || runes[i] == '！' || runes[i] == '？' ||
						runes[i] == '.' || runes[i] == '!' || runes[i] == '?' ||
						runes[i] == '\n' {
						bestBreak = i + 1
						break
					}
				}
			}
			if bestBreak > 0 {
				end = bestBreak
			}
		}

		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			chunks = append(chunks, chunk)
		}

		// 确保 start 始终前进，避免死循环
		newStart := end - overlap
		if newStart <= start {
			newStart = end
		}
		start = newStart
		if start >= len(runes) {
			break
		}
	}

	return chunks
}
