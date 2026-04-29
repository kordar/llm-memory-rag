module github.com/kordar/llm-memory-rag

go 1.25.5

require (
	github.com/google/uuid v1.6.0
	github.com/kordar/llm-memory v0.0.0-00010101000000-000000000000
	github.com/kordar/llm-rag v0.0.0-00010101000000-000000000000
	github.com/lib/pq v1.12.0
	github.com/pgvector/pgvector-go v0.3.0
)

replace github.com/kordar/llm-memory => ../llm-memory

replace github.com/kordar/llm-rag => ../llm-rag
