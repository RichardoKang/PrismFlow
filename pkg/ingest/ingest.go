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
	"time"

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

	// 初始化 Embedding Adapter
	var embedAdapter ports.EmbeddingProvider
	if cfg.Embedding.Provider == "ollama" {
		embedAdapter = embedding.NewOllamaEmbeddingAdapter(embedding.OllamaConfig{
			BaseURL: cfg.Embedding.BaseURL,
			Model:   cfg.Embedding.Model,
		})
		log.Printf("使用 Ollama embedding: %s", cfg.Embedding.Model)
	} else {
		log.Fatal("请配置 Ollama embedding provider")
	}

	// 读取并分割 Markdown 文件
	log.Printf("正在读取文件: %s", *mdPath)
	docs, err := splitMarkdown(*mdPath, *chunkSize, *overlap)
	if err != nil {
		log.Fatalf("分割文档失败: %v", err)
	}
	log.Printf("共分割为 %d 个文档块", len(docs))

	// 导入到 Milvus
	ctx := context.Background()
	successCount := 0
	failCount := 0

	for i, doc := range docs {
		// 生成 embedding
		vector, err := embedAdapter.Embed(ctx, doc.Content)
		if err != nil {
			log.Printf("[%d/%d] 生成 embedding 失败: %v", i+1, len(docs), err)
			failCount++
			continue
		}

		// 存入 Milvus
		meta := map[string]interface{}{
			"title":  doc.Title,
			"source": doc.Source,
		}
		err = milvusAdapter.Store(ctx, fmt.Sprintf("doc_%d", i), vector, doc.Content, meta)
		if err != nil {
			log.Printf("[%d/%d] 存入 Milvus 失败: %v", i+1, len(docs), err)
			failCount++
			continue
		}

		successCount++
		if successCount%10 == 0 {
			log.Printf("进度: %d/%d (成功: %d, 失败: %d)", i+1, len(docs), successCount, failCount)
		}

		// 避免请求过快
		time.Sleep(100 * time.Millisecond)
	}

	log.Printf("导入完成! 成功: %d, 失败: %d, 总计: %d", successCount, failCount, len(docs))
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

	// 标题正则表达式
	titleRegex := regexp.MustCompile(`^#+\s+(.+)$`)

	scanner := bufio.NewScanner(file)
	// 增大缓冲区以处理长行
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	fileName := filePath[strings.LastIndex(filePath, "/")+1:]

	for scanner.Scan() {
		line := scanner.Text()

		// 检测标题行
		if matches := titleRegex.FindStringSubmatch(line); len(matches) > 1 {
			// 保存之前的内容
			if currentContent.Len() > 0 {
				chunks := splitIntoChunks(currentContent.String(), chunkSize, overlap)
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
			// 开始新章节
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
		chunks := splitIntoChunks(currentContent.String(), chunkSize, overlap)
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

// splitIntoChunks 将文本分割成指定大小的块，带重叠
func splitIntoChunks(text string, chunkSize, overlap int) []string {
	text = strings.TrimSpace(text)
	if len(text) == 0 {
		return nil
	}

	// 如果文本小于 chunkSize，直接返回
	if len(text) <= chunkSize {
		return []string{text}
	}

	var chunks []string
	runes := []rune(text) // 使用 rune 处理中文

	start := 0
	for start < len(runes) {
		end := start + chunkSize
		if end > len(runes) {
			end = len(runes)
		}

		// 尝试在句子结尾处断开
		if end < len(runes) {
			// 向后查找句子结束符
			for i := end; i > start+chunkSize/2; i-- {
				if runes[i] == '。' || runes[i] == '！' || runes[i] == '？' ||
					runes[i] == '\n' || runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
					end = i + 1
					break
				}
			}
		}

		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			chunks = append(chunks, chunk)
		}

		// 计算下一个起始位置（考虑重叠）
		start = end - overlap
		if start <= 0 || start >= len(runes) {
			break
		}
	}

	return chunks
}
