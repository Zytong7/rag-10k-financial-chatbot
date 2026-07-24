"""
Batch-runs a golden set of questions against one config and writes a CSV of
answers + retrieved sources. Run this once per config (LLM x embedding x
chunk size x k) and stack the CSVs into the config x question matrix.

Usage (from the project root):
    python app/eval_harness.py \\
        --questions data/golden_set.csv \\
        --llm gemini-3.5-flash \\
        --embedding-provider gemini \\
        --chunk-size 1500 --chunk-overlap 300 --k 4 \\
        --out results/gemini_1500_k4.csv

golden_set.csv must have at least columns: id,question
An optional expected_answer column is passed through untouched so it's easy
to eyeball answer vs. expected side by side.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import (
    ALL_LLM_MODELS,
    GEMINI_EMBEDDING_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    RAGConfig,
)
from ingest import build_vector_store
from retrieval import answer_question


def parse_args():
    p = argparse.ArgumentParser(description="Run a golden set against one RAG config.")
    p.add_argument("--questions", required=True, help="Path to golden_set.csv")
    p.add_argument("--out", required=True, help="Path to write results CSV")
    p.add_argument("--llm", default="gemini-3.5-flash", choices=ALL_LLM_MODELS)
    p.add_argument("--embedding-provider", choices=["gemini", "openai", "ollama"], default="gemini")
    p.add_argument("--chunk-size", type=int, default=2500)
    p.add_argument("--chunk-overlap", type=int, default=500)
    p.add_argument("--k", type=int, default=6, help="Chunks per company")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--rebuild-index", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    from config import EMBEDDING_MODELS
    embedding_model = EMBEDDING_MODELS[args.embedding_provider]

    cfg = RAGConfig(
        llm_model=args.llm,
        embedding_provider=args.embedding_provider,
        embedding_model=embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        temperature=args.temperature,
        k_per_company=args.k,
    )

    questions_path = Path(args.questions)
    if not questions_path.exists():
        sys.exit(f"Questions file not found: {questions_path}")

    with open(questions_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("Questions file is empty.")

    print(f"Loading/building vector store for config: {cfg.index_key()} ...")
    vector_store = build_vector_store(cfg, force_rebuild=args.rebuild_index)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "question",
        "expected_answer",
        "answer",
        "error",
        "companies_searched",
        "sources",
        "latency_sec",
        "llm",
        "embedding_provider",
        "chunk_size",
        "chunk_overlap",
        "k_per_company",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            question = row["question"]
            print(f"  [{row.get('id', '?')}] {question[:70]}...")
            start = time.time()
            answer, error, sources, searched = "", "", "", ""
            try:
                result = answer_question(vector_store, cfg, question)
                answer = result.text
                searched = ", ".join(result.companies_searched)
                sources = "; ".join(
                    f"{d.metadata.get('company')} p.{d.metadata.get('page')}"
                    for d in result.sources
                )
            except Exception as e:
                error = str(e)
            elapsed = round(time.time() - start, 2)

            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "question": question,
                    "expected_answer": row.get("expected_answer", ""),
                    "answer": answer,
                    "error": error,
                    "companies_searched": searched,
                    "sources": sources,
                    "latency_sec": elapsed,
                    "llm": cfg.llm_model,
                    "embedding_provider": cfg.embedding_provider,
                    "chunk_size": cfg.chunk_size,
                    "chunk_overlap": cfg.chunk_overlap,
                    "k_per_company": cfg.k_per_company,
                }
            )

    print(f"Done. Wrote {len(rows)} results to {out_path}")


if __name__ == "__main__":
    main()
