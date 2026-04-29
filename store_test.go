package memoryrag

import (
	"context"
	"errors"
	"testing"
	"time"

	llmmemory "github.com/kordar/llm-memory"
)

type fakeReranker struct {
	scores []float32
	err    error
}

func (f fakeReranker) Rerank(context.Context, string, []string) ([]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.scores, nil
}

func TestNormalizeConfig_Defaults(t *testing.T) {
	got := normalizeConfig(Config{})
	if got.Table != "memory_records" {
		t.Fatalf("unexpected table: %s", got.Table)
	}
	if got.Dimension != 1024 {
		t.Fatalf("unexpected dimension: %d", got.Dimension)
	}
	if got.RecallMultiplier != 4 {
		t.Fatalf("unexpected recall multiplier: %d", got.RecallMultiplier)
	}
}

func TestIsSafeIdentifier(t *testing.T) {
	if !isSafeIdentifier("memory_records_1") {
		t.Fatal("expected valid identifier")
	}
	if isSafeIdentifier("1bad") {
		t.Fatal("expected invalid identifier")
	}
	if isSafeIdentifier("bad-name") {
		t.Fatal("expected invalid identifier with dash")
	}
}

func TestNormalizeFloat32(t *testing.T) {
	got := normalizeFloat32([]float32{10, 20, 30})
	if len(got) != 3 {
		t.Fatalf("unexpected length: %d", len(got))
	}
	if got[0] >= got[1] || got[1] >= got[2] {
		t.Fatalf("expected ascending normalized scores: %#v", got)
	}

	got2 := normalizeFloat32([]float32{5, 5})
	if got2[0] != 1 || got2[1] != 1 {
		t.Fatalf("expected all ones for equal inputs, got %#v", got2)
	}
}

func TestFilterAndTrim(t *testing.T) {
	in := []llmmemory.Record{
		{ID: "a", Score: 0.9},
		{ID: "b", Score: 0.3},
		{ID: "c", Score: 0.7},
	}
	got := filterAndTrim(in, 0.5, 1)
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("unexpected filter result: %#v", got)
	}
}

func TestApplyRerank(t *testing.T) {
	now := time.Now()
	store := &PGVectorMemoryStore{
		reranker: fakeReranker{scores: []float32{0.1, 0.9}},
	}
	in := []llmmemory.Record{
		{ID: "a", Content: "doc-a", Score: 0.8, CreatedAt: now.Add(-time.Hour)},
		{ID: "b", Content: "doc-b", Score: 0.6, CreatedAt: now},
	}
	out := store.applyRerank(context.Background(), "query", in)
	if len(out) != 2 {
		t.Fatalf("unexpected output length: %d", len(out))
	}
	if out[0].ID != "b" {
		t.Fatalf("expected rerank promote b first, got %#v", out)
	}
}

func TestApplyRerank_DegradeOnError(t *testing.T) {
	store := &PGVectorMemoryStore{
		reranker: fakeReranker{err: errors.New("boom")},
	}
	in := []llmmemory.Record{
		{ID: "a", Content: "doc-a", Score: 0.8},
	}
	out := store.applyRerank(context.Background(), "query", in)
	if len(out) != 1 || out[0].ID != "a" {
		t.Fatalf("expected fallback output, got %#v", out)
	}
}
