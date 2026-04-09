# PrismFlow RAG 评估方法

## 评估框架

评估脚本位于 `eval/eval.go`，通过对比不同检索策略的效果来量化 RAG 系统的召回质量。

## 运行方式

```bash
go run eval/eval.go \
  -config configs/config.yaml \
  -testset eval/testset.json \
  -output eval/results
```

## 测试集格式

`eval/testset.json`：

```json
[
  {
    "query": "MySQL 中什么是事务隔离级别？",
    "expected_answer": "MySQL 支持四种事务隔离级别...",
    "ground_truth_context": "事务隔离级别"
  }
]
```

## 评估指标

| 指标 | 说明 |
|------|------|
| Context Precision | 检索结果中相关文档的比例（前 K 个结果中有多少是相关的） |
| Context Recall | 相关文档被检索到的比例（所有相关文档中有多少被召回） |
| Avg Latency | 平均检索延迟 |

## 实验对比

评估框架支持三种检索策略的自动对比：

| 实验 | 策略 | 说明 |
|------|------|------|
| baseline | 纯向量检索 | Milvus HNSW + COSINE |
| +Rerank | 向量检索 + Rerank | 加入 BGE-Reranker 精排 |
| +Hybrid | Hybrid Search + Rerank | 向量 + BM25 RRF 融合 + 精排 |

## 输出

评估结果保存在 `eval/results/` 目录：

- `baseline_summary.json`：纯向量检索的指标
- `rerank_summary.json`：+Rerank 的指标
- `hybrid_summary.json`：+Hybrid 的指标

## 扩展

如需更完整的评估（Faithfulness、Answer Relevancy），可接入 RAGAS 框架或使用 LLM-as-Judge 方式评估生成质量。当前评估聚焦于检索阶段的效果对比。
