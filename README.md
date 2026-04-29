# llm-memory-rag

`llm-memory-rag` 提供一个面向生产的向量 `Store` 实现，直接兼容 `github.com/kordar/llm-memory` 的 `Store` 接口：

- 向量库：PostgreSQL + pgvector
- 向量化：复用 `llm-rag/embedding` 客户端
- 精排：可选 `llm-rag/rerank` 客户端（失败自动降级）
- 接口：`Recall/Write/Update/Delete`

## 安装

```bash
go get github.com/kordar/llm-memory-rag
```

## 快速开始

```go
package main

import (
	"context"

	llmmemory "github.com/kordar/llm-memory"
	memoryrag "github.com/kordar/llm-memory-rag"
	"github.com/kordar/llm-rag/embedding"
	"github.com/kordar/llm-rag/rerank"
)

func main() {
	embedder := embedding.New(
		embedding.NewVLLMProvider("http://localhost:8000"),
		embedding.WithModel("bge-large-zh"),
	)
	reranker := rerank.New("http://localhost:8001", "bge-reranker")

	store, err := memoryrag.NewWithRAGClients(
		"postgres://user:pass@localhost:5432/rag?sslmode=disable",
		embedder,
		reranker,
		memoryrag.Config{
			Table:     "memory_records",
			Dimension: 1024,
		},
	)
	if err != nil {
		panic(err)
	}
	defer store.Close()

	_ = store.Write(context.Background(), llmmemory.Record{
		UserID:  "u1",
		Content: "报销流程是提交申请后审批打款",
		Type:    llmmemory.RecordSemantic,
		Score:   0.9,
	})

	_, _ = store.Recall(context.Background(), "u1", "报销怎么走", llmmemory.WithTopK(5))
}
```

## 表结构

库会自动创建扩展和表：

- `CREATE EXTENSION IF NOT EXISTS vector`
- `memory_records`（可配置）
- 向量列：`embedding VECTOR(<dimension>)`
- 索引：`user_id` BTree + `embedding` ivfflat(cosine)

## 说明

- `Write` 使用 upsert（`ON CONFLICT(id)`）。
- `Recall` 先向量召回，再可选 rerank 融合排序。
- `Update` 会重新向量化内容并更新向量。
