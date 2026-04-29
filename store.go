package memoryrag

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	llmmemory "github.com/kordar/llm-memory"
	"github.com/kordar/llm-rag/embedding"
	"github.com/kordar/llm-rag/rerank"
	"github.com/lib/pq"
	"github.com/pgvector/pgvector-go"
)

// Embedder is compatible with llm-rag embedding client.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Reranker is compatible with llm-rag rerank client.
type Reranker interface {
	Rerank(ctx context.Context, query string, texts []string) ([]float32, error)
}

type Config struct {
	Table            string
	Dimension        int
	RecallMultiplier int
}

type PGVectorMemoryStore struct {
	db               *sql.DB
	table            string
	dimension        int
	recallMultiplier int
	embedder         Embedder
	reranker         Reranker
}

func NewWithRAGClients(dsn string, embedder *embedding.Client, reranker *rerank.Client, cfg Config) (*PGVectorMemoryStore, error) {
	return NewPGVectorMemoryStore(dsn, embedder, reranker, cfg)
}

func NewPGVectorMemoryStore(dsn string, embedder Embedder, reranker Reranker, cfg Config) (*PGVectorMemoryStore, error) {
	if embedder == nil {
		return nil, errors.New("memory-rag: nil embedder")
	}
	cfg = normalizeConfig(cfg)
	if !isSafeIdentifier(cfg.Table) {
		return nil, fmt.Errorf("memory-rag: invalid table name: %s", cfg.Table)
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, err
	}
	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, err
	}

	s := &PGVectorMemoryStore{
		db:               db,
		table:            cfg.Table,
		dimension:        cfg.Dimension,
		recallMultiplier: cfg.RecallMultiplier,
		embedder:         embedder,
		reranker:         reranker,
	}
	if err := s.ensureSchema(context.Background()); err != nil {
		_ = db.Close()
		return nil, err
	}
	return s, nil
}

func (s *PGVectorMemoryStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *PGVectorMemoryStore) Recall(ctx context.Context, userID string, query string, opts ...llmmemory.Option) ([]llmmemory.Record, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("memory-rag: nil store")
	}
	userID = strings.TrimSpace(userID)
	query = strings.TrimSpace(query)
	if userID == "" || query == "" {
		return []llmmemory.Record{}, nil
	}

	recallOpt := llmmemory.BuildRecallOption(opts...)
	if recallOpt.TopK <= 0 {
		return []llmmemory.Record{}, nil
	}

	embs, err := s.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(embs) != 1 {
		return nil, errors.New("memory-rag: invalid embedding response")
	}

	recallTop := recallOpt.TopK * s.recallMultiplier
	if recallTop < recallOpt.TopK {
		recallTop = recallOpt.TopK
	}

	sqlQuery := fmt.Sprintf(`
SELECT id, content, rec_type, tags, base_score, created_at, updated_at, (1 - (embedding <=> $1)) AS similarity
FROM %s
WHERE user_id = $2
ORDER BY embedding <=> $1
LIMIT $3
`, s.table)
	rows, err := s.db.QueryContext(ctx, sqlQuery, pgvector.NewVector(embs[0]), userID, recallTop)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	records := make([]llmmemory.Record, 0, recallTop)
	for rows.Next() {
		var r llmmemory.Record
		var typ string
		var tags pq.StringArray
		var baseScore float64
		var similarity float64
		if err := rows.Scan(&r.ID, &r.Content, &typ, &tags, &baseScore, &r.CreatedAt, &r.UpdatedAt, &similarity); err != nil {
			return nil, err
		}
		r.UserID = userID
		r.Type = llmmemory.RecordType(typ)
		r.Tags = append([]string(nil), tags...)
		r.Score = blendScore(similarity, baseScore)
		records = append(records, r)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	records = s.applyRerank(ctx, query, records)
	records = filterAndTrim(records, recallOpt.MinScore, recallOpt.TopK)
	return records, nil
}

func (s *PGVectorMemoryStore) Write(ctx context.Context, record llmmemory.Record) error {
	if s == nil || s.db == nil {
		return errors.New("memory-rag: nil store")
	}
	record.UserID = strings.TrimSpace(record.UserID)
	record.Content = strings.TrimSpace(record.Content)
	if record.UserID == "" {
		return errors.New("memory-rag: empty user id")
	}
	if record.Content == "" {
		return nil
	}
	if record.ID == "" {
		record.ID = uuid.NewString()
	}
	if record.Type == "" {
		record.Type = llmmemory.RecordSemantic
	}
	now := time.Now()
	if record.CreatedAt.IsZero() {
		record.CreatedAt = now
	}
	if record.UpdatedAt.IsZero() {
		record.UpdatedAt = now
	}

	embs, err := s.embedder.Embed(ctx, []string{record.Content})
	if err != nil {
		return err
	}
	if len(embs) != 1 {
		return errors.New("memory-rag: invalid embedding response")
	}

	sqlQuery := fmt.Sprintf(`
INSERT INTO %s (id, user_id, content, embedding, rec_type, tags, base_score, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (id)
DO UPDATE SET
  user_id = EXCLUDED.user_id,
  content = EXCLUDED.content,
  embedding = EXCLUDED.embedding,
  rec_type = EXCLUDED.rec_type,
  tags = EXCLUDED.tags,
  base_score = EXCLUDED.base_score,
  created_at = EXCLUDED.created_at,
  updated_at = EXCLUDED.updated_at
`, s.table)

	_, err = s.db.ExecContext(ctx, sqlQuery,
		record.ID,
		record.UserID,
		record.Content,
		pgvector.NewVector(embs[0]),
		string(record.Type),
		pq.Array(record.Tags),
		record.Score,
		record.CreatedAt,
		record.UpdatedAt,
	)
	return err
}

func (s *PGVectorMemoryStore) Update(ctx context.Context, id string, record llmmemory.Record) error {
	if s == nil || s.db == nil {
		return errors.New("memory-rag: nil store")
	}
	id = strings.TrimSpace(id)
	record.UserID = strings.TrimSpace(record.UserID)
	record.Content = strings.TrimSpace(record.Content)
	if id == "" || record.UserID == "" {
		return errors.New("memory-rag: invalid update args")
	}
	if record.Content == "" {
		return errors.New("memory-rag: empty content")
	}
	if record.Type == "" {
		record.Type = llmmemory.RecordSemantic
	}

	if record.CreatedAt.IsZero() {
		fetchQuery := fmt.Sprintf(`SELECT created_at FROM %s WHERE id = $1`, s.table)
		if err := s.db.QueryRowContext(ctx, fetchQuery, id).Scan(&record.CreatedAt); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return errors.New("memory-rag: record not found")
			}
			return err
		}
	}
	if record.UpdatedAt.IsZero() {
		record.UpdatedAt = time.Now()
	}

	embs, err := s.embedder.Embed(ctx, []string{record.Content})
	if err != nil {
		return err
	}
	if len(embs) != 1 {
		return errors.New("memory-rag: invalid embedding response")
	}

	updateQuery := fmt.Sprintf(`
UPDATE %s
SET user_id = $2,
    content = $3,
    embedding = $4,
    rec_type = $5,
    tags = $6,
    base_score = $7,
    created_at = $8,
    updated_at = $9
WHERE id = $1
`, s.table)
	res, err := s.db.ExecContext(ctx, updateQuery,
		id,
		record.UserID,
		record.Content,
		pgvector.NewVector(embs[0]),
		string(record.Type),
		pq.Array(record.Tags),
		record.Score,
		record.CreatedAt,
		record.UpdatedAt,
	)
	if err != nil {
		return err
	}
	affected, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if affected == 0 {
		return errors.New("memory-rag: record not found")
	}
	return nil
}

func (s *PGVectorMemoryStore) Delete(ctx context.Context, id string) error {
	if s == nil || s.db == nil {
		return errors.New("memory-rag: nil store")
	}
	id = strings.TrimSpace(id)
	if id == "" {
		return nil
	}
	deleteQuery := fmt.Sprintf(`DELETE FROM %s WHERE id = $1`, s.table)
	_, err := s.db.ExecContext(ctx, deleteQuery, id)
	return err
}

func (s *PGVectorMemoryStore) ensureSchema(ctx context.Context) error {
	createExt := `CREATE EXTENSION IF NOT EXISTS vector`
	if _, err := s.db.ExecContext(ctx, createExt); err != nil {
		return err
	}

	createTable := fmt.Sprintf(`
CREATE TABLE IF NOT EXISTS %s (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(%d) NOT NULL,
  rec_type TEXT NOT NULL,
  tags TEXT[] NOT NULL DEFAULT '{}',
  base_score DOUBLE PRECISION NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL
)`, s.table, s.dimension)
	if _, err := s.db.ExecContext(ctx, createTable); err != nil {
		return err
	}

	userIdx := fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_user_id_idx ON %s (user_id)`, s.table, s.table)
	if _, err := s.db.ExecContext(ctx, userIdx); err != nil {
		return err
	}
	vecIdx := fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_embedding_ivfflat ON %s USING ivfflat (embedding vector_cosine_ops)`, s.table, s.table)
	if _, err := s.db.ExecContext(ctx, vecIdx); err != nil {
		return err
	}
	return nil
}

func (s *PGVectorMemoryStore) applyRerank(ctx context.Context, query string, records []llmmemory.Record) []llmmemory.Record {
	if s == nil || s.reranker == nil || len(records) == 0 {
		return records
	}
	texts := make([]string, 0, len(records))
	for _, r := range records {
		texts = append(texts, r.Content)
	}
	scores, err := s.reranker.Rerank(ctx, query, texts)
	if err != nil || len(scores) != len(records) {
		return records
	}
	norm := normalizeFloat32(scores)
	for i := range records {
		records[i].Score = 0.6*records[i].Score + 0.4*norm[i]
	}
	sort.SliceStable(records, func(i, j int) bool {
		if records[i].Score == records[j].Score {
			return records[i].CreatedAt.After(records[j].CreatedAt)
		}
		return records[i].Score > records[j].Score
	})
	return records
}

func filterAndTrim(records []llmmemory.Record, minScore float64, topK int) []llmmemory.Record {
	filtered := records[:0]
	for _, r := range records {
		if r.Score >= minScore {
			filtered = append(filtered, r)
		}
	}
	if topK <= 0 || topK >= len(filtered) {
		out := make([]llmmemory.Record, len(filtered))
		copy(out, filtered)
		return out
	}
	out := make([]llmmemory.Record, topK)
	copy(out, filtered[:topK])
	return out
}

func normalizeConfig(cfg Config) Config {
	if strings.TrimSpace(cfg.Table) == "" {
		cfg.Table = "memory_records"
	}
	if cfg.Dimension <= 0 {
		cfg.Dimension = 1024
	}
	if cfg.RecallMultiplier <= 0 {
		cfg.RecallMultiplier = 4
	}
	return cfg
}

func blendScore(similarity float64, baseScore float64) float64 {
	return 0.7*similarity + 0.3*baseScore
}

func normalizeFloat32(scores []float32) []float64 {
	if len(scores) == 0 {
		return []float64{}
	}
	minV := scores[0]
	maxV := scores[0]
	for _, s := range scores[1:] {
		if s < minV {
			minV = s
		}
		if s > maxV {
			maxV = s
		}
	}
	out := make([]float64, len(scores))
	if maxV == minV {
		for i := range out {
			out[i] = 1
		}
		return out
	}
	denom := float64(maxV - minV)
	for i, s := range scores {
		out[i] = float64(s-minV) / denom
	}
	return out
}

func isSafeIdentifier(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		if i == 0 {
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_' {
				continue
			}
			return false
		}
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			continue
		}
		return false
	}
	return true
}
