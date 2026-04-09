package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"PrismFlow/internal/adapter/embedding"
	"PrismFlow/internal/adapter/reranker"
	"PrismFlow/internal/adapter/retriever"
	"PrismFlow/internal/adapter/vectordb"
	"PrismFlow/internal/config"
	"PrismFlow/internal/core/ports"
)

// TestCase 评估测试用例
type TestCase struct {
	Query              string `json:"query"`
	ExpectedAnswer     string `json:"expected_answer"`
	GroundTruthContext string `json:"ground_truth_context"`
}

// EvalResult 单条评估结果
type EvalResult struct {
	Query            string   `json:"query"`
	RetrievedDocs    []string `json:"retrieved_docs"`
	ContextPrecision float64  `json:"context_precision"`
	ContextRecall    float64  `json:"context_recall"`
	Mode             string   `json:"mode"`
}

// EvalSummary 评估汇总
type EvalSummary struct {
	Mode             string  `json:"mode"`
	AvgPrecision     float64 `json:"avg_context_precision"`
	AvgRecall        float64 `json:"avg_context_recall"`
	TotalQueries     int     `json:"total_queries"`
	AvgLatencyMs     float64 `json:"avg_latency_ms"`
}

func main() {
	configPath := flag.String("config", "configs/config.yaml", "配置文件路径")
	testsetPath := flag.String("testset", "eval/testset.json", "测试集路径")
	outputDir := flag.String("output", "eval/results", "结果输出目录")
	flag.Parse()

	// 加载配置
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("加载配置失败: %v", err)
	}

	// 初始化组件
	milvusAdapter, err := vectordb.NewMilvusAdapter(vectordb.MilvusConfig{
		Address:        cfg.VectorDBAddr(),
		CollectionName: cfg.VectorDB.CollectionName,
		Dimension:      cfg.VectorDB.Dimension,
	})
	if err != nil {
		log.Fatalf("连接 Milvus 失败: %v", err)
	}
	defer milvusAdapter.Close()

	var embedAdapter ports.EmbeddingProvider
	if cfg.Embedding.Provider == "ollama" {
		embedAdapter = embedding.NewOllamaEmbeddingAdapter(embedding.OllamaConfig{
			BaseURL: cfg.Embedding.BaseURL,
			Model:   cfg.Embedding.Model,
		})
	} else {
		log.Fatal("请配置 Ollama embedding provider")
	}

	// 加载测试集
	testCases, err := loadTestSet(*testsetPath)
	if err != nil {
		log.Fatalf("加载测试集失败: %v", err)
	}
	log.Printf("加载 %d 条测试用例", len(testCases))

	topK := cfg.RAG.TopK
	ctx := context.Background()

	// ========== 实验 1: 纯向量检索（基线） ==========
	log.Println("=== 实验 1: 纯向量检索 ===")
	baselineResults, baselineSummary := runExperiment(ctx, testCases, embedAdapter, func(ctx context.Context, query string, vector []float32) ([]string, error) {
		results, err := milvusAdapter.Search(ctx, vector, topK)
		if err != nil {
			return nil, err
		}
		docs := make([]string, len(results))
		for i, r := range results {
			docs[i] = r.Content
		}
		return docs, nil
	}, "baseline_vector")

	// ========== 实验 2: 向量检索 + Rerank ==========
	log.Println("=== 实验 2: 向量检索 + Rerank ===")
	var rerankerAdapter ports.RerankerProvider
	if cfg.Reranker.Provider == "bge" {
		rerankerAdapter = reranker.NewBGEReranker(reranker.BGEConfig{
			BaseURL: cfg.Reranker.BaseURL,
			Model:   cfg.Reranker.Model,
		}, nil)
	} else {
		rerankerAdapter = reranker.NewMockReranker()
	}
	rerankTopN := cfg.Reranker.TopN

	rerankResults, rerankSummary := runExperiment(ctx, testCases, embedAdapter, func(ctx context.Context, query string, vector []float32) ([]string, error) {
		results, err := milvusAdapter.Search(ctx, vector, topK)
		if err != nil {
			return nil, err
		}
		reranked, err := rerankerAdapter.Rerank(ctx, query, results, rerankTopN)
		if err != nil {
			return nil, err
		}
		docs := make([]string, len(reranked))
		for i, r := range reranked {
			docs[i] = r.Content
		}
		return docs, nil
	}, "vector_rerank")

	// ========== 实验 3: Hybrid Search + Rerank ==========
	log.Println("=== 实验 3: Hybrid Search + Rerank ===")
	hybridRetriever := retriever.NewHybridRetriever(milvusAdapter, nil, nil)

	hybridResults, hybridSummary := runExperiment(ctx, testCases, embedAdapter, func(ctx context.Context, query string, vector []float32) ([]string, error) {
		results, err := hybridRetriever.Search(ctx, query, vector, topK)
		if err != nil {
			return nil, err
		}
		reranked, err := rerankerAdapter.Rerank(ctx, query, results, rerankTopN)
		if err != nil {
			return nil, err
		}
		docs := make([]string, len(reranked))
		for i, r := range reranked {
			docs[i] = r.Content
		}
		return docs, nil
	}, "hybrid_rerank")

	// 输出结果
	saveResults(*outputDir, "baseline", baselineResults, baselineSummary)
	saveResults(*outputDir, "rerank", rerankResults, rerankSummary)
	saveResults(*outputDir, "hybrid", hybridResults, hybridSummary)

	// 打印对比
	fmt.Println("\n========== 评估结果对比 ==========")
	fmt.Printf("%-20s %-18s %-18s %-15s\n", "Mode", "Avg Precision", "Avg Recall", "Avg Latency(ms)")
	fmt.Printf("%-20s %-18.4f %-18.4f %-15.2f\n", baselineSummary.Mode, baselineSummary.AvgPrecision, baselineSummary.AvgRecall, baselineSummary.AvgLatencyMs)
	fmt.Printf("%-20s %-18.4f %-18.4f %-15.2f\n", rerankSummary.Mode, rerankSummary.AvgPrecision, rerankSummary.AvgRecall, rerankSummary.AvgLatencyMs)
	fmt.Printf("%-20s %-18.4f %-18.4f %-15.2f\n", hybridSummary.Mode, hybridSummary.AvgPrecision, hybridSummary.AvgRecall, hybridSummary.AvgLatencyMs)
}

type searchFunc func(ctx context.Context, query string, vector []float32) ([]string, error)

func runExperiment(ctx context.Context, testCases []TestCase, embedder ports.EmbeddingProvider, search searchFunc, mode string) ([]EvalResult, EvalSummary) {
	var results []EvalResult
	var totalPrecision, totalRecall float64
	var totalLatency time.Duration

	for i, tc := range testCases {
		start := time.Now()

		// Embedding
		vector, err := embedder.Embed(ctx, tc.Query)
		if err != nil {
			log.Printf("[%d] embedding 失败: %v", i+1, err)
			continue
		}

		// 检索
		docs, err := search(ctx, tc.Query, vector)
		if err != nil {
			log.Printf("[%d] 检索失败: %v", i+1, err)
			continue
		}

		latency := time.Since(start)
		totalLatency += latency

		// 计算 Context Precision: 检索到的文档中有多少包含 ground truth 关键词
		precision := contextPrecision(docs, tc.GroundTruthContext)
		// 计算 Context Recall: ground truth 关键词是否被检索到
		recall := contextRecall(docs, tc.GroundTruthContext)

		totalPrecision += precision
		totalRecall += recall

		results = append(results, EvalResult{
			Query:            tc.Query,
			RetrievedDocs:    docs,
			ContextPrecision: precision,
			ContextRecall:    recall,
			Mode:             mode,
		})

		log.Printf("[%s][%d/%d] precision=%.4f recall=%.4f latency=%dms",
			mode, i+1, len(testCases), precision, recall, latency.Milliseconds())
	}

	n := float64(len(results))
	if n == 0 {
		n = 1
	}

	summary := EvalSummary{
		Mode:         mode,
		AvgPrecision: totalPrecision / n,
		AvgRecall:    totalRecall / n,
		TotalQueries: len(results),
		AvgLatencyMs: float64(totalLatency.Milliseconds()) / n,
	}

	return results, summary
}

// contextPrecision 计算检索到的文档中包含 ground truth 关键词的比例
func contextPrecision(docs []string, groundTruth string) float64 {
	if len(docs) == 0 {
		return 0
	}
	keywords := strings.Fields(groundTruth)
	relevant := 0
	for _, doc := range docs {
		for _, kw := range keywords {
			if strings.Contains(doc, kw) {
				relevant++
				break
			}
		}
	}
	return float64(relevant) / float64(len(docs))
}

// contextRecall 计算 ground truth 关键词被检索到的比例
func contextRecall(docs []string, groundTruth string) float64 {
	keywords := strings.Fields(groundTruth)
	if len(keywords) == 0 {
		return 0
	}
	allText := strings.Join(docs, " ")
	found := 0
	for _, kw := range keywords {
		if strings.Contains(allText, kw) {
			found++
		}
	}
	return float64(found) / float64(len(keywords))
}

func loadTestSet(path string) ([]TestCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cases []TestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, err
	}
	return cases, nil
}

func saveResults(dir, name string, results []EvalResult, summary EvalSummary) {
	// 保存详细结果
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile(fmt.Sprintf("%s/%s_results.json", dir, name), data, 0644)

	// 保存汇总
	summaryData, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile(fmt.Sprintf("%s/%s_summary.json", dir, name), summaryData, 0644)
}
