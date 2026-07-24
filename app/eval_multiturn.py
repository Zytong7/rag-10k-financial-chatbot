"""
Multi-turn evaluation harness. eval_harness.py treats every question as a
fresh session; this one replays scripted conversations, feeding each turn the
accumulated chat history via the `messages` parameter that
retrieval.answer_question() already supports (but the single-turn harness
never used).

Why this matters for the robustness tests: retrieval embeds ONLY the current
question (by design -- see retrieval.retrieve). So a follow-up like "and its
operating margin?" carries no company name into the retriever. Whether the
pipeline survives pronouns/ellipsis across turns is exactly what this harness
measures.

Input CSV columns: conv_id, turn, question, expected_answer (optional),
category (optional). Turns are replayed in ascending `turn` order per conv_id.

Usage (from project root):
    python app/eval_multiturn.py --conversations data/multiturn_set.csv \\
        --llm claude-opus-4-8 --embedding-provider openai \\
        --out results/multiturn_claude.csv
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import ALL_LLM_MODELS, EMBEDDING_MODELS, RAGConfig  # noqa: E402
from ingest import build_vector_store  # noqa: E402
from retrieval import answer_question  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Replay scripted multi-turn conversations.")
    p.add_argument("--conversations", required=True,
                   help="CSV with conv_id,turn,question[,expected_answer,category]")
    p.add_argument("--out", required=True)
    p.add_argument("--llm", default="claude-opus-4-8", choices=ALL_LLM_MODELS)
    p.add_argument("--embedding-provider", choices=["gemini", "openai", "ollama"],
                   default="openai")
    p.add_argument("--chunk-size", type=int, default=2500)
    p.add_argument("--chunk-overlap", type=int, default=500)
    p.add_argument("--k", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = RAGConfig(
        llm_model=args.llm,
        embedding_provider=args.embedding_provider,
        embedding_model=EMBEDDING_MODELS[args.embedding_provider],
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k_per_company=args.k,
    )

    path = Path(args.conversations)
    if not path.exists():
        sys.exit(f"Conversations file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("Conversations file is empty.")

    convs = defaultdict(list)
    for r in rows:
        convs[r["conv_id"]].append(r)
    for turns in convs.values():
        turns.sort(key=lambda r: int(r["turn"]))

    print(f"Loading/building vector store for config: {cfg.index_key()} ...")
    vector_store = build_vector_store(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "conv_id", "turn", "question", "expected_answer", "category",
        "answer", "error", "companies_searched", "sources", "latency_sec",
        "llm", "embedding_provider", "chunk_size", "chunk_overlap", "k_per_company",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for conv_id, turns in convs.items():
            print(f"\n=== conversation {conv_id} ({len(turns)} turns) ===")
            messages = []  # accumulated history in app.py's message format
            for r in turns:
                question = r["question"]
                print(f"  [turn {r['turn']}] {question[:70]}")
                start = time.time()
                answer, error, sources, searched = "", "", "", ""
                try:
                    result = answer_question(vector_store, cfg, question,
                                             messages=messages)
                    answer = result.text
                    searched = ", ".join(result.companies_searched)
                    sources = "; ".join(
                        f"{d.metadata.get('company')} p.{d.metadata.get('page')}"
                        for d in result.sources
                    )
                except Exception as e:
                    error = str(e)
                elapsed = round(time.time() - start, 2)

                writer.writerow({
                    "conv_id": conv_id,
                    "turn": r["turn"],
                    "question": question,
                    "expected_answer": r.get("expected_answer", ""),
                    "category": r.get("category", ""),
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
                })

                # Feed this turn into the next turn's history. On an API error
                # we still record the user turn so the transcript stays honest.
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant",
                                 "content": answer or f"(error: {error})"})

    print(f"\nDone. Wrote {sum(len(t) for t in convs.values())} turns to {out_path}")


if __name__ == "__main__":
    main()
