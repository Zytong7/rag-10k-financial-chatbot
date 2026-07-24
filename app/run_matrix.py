"""
Orchestrates the full ablation matrix by shelling out to eval_harness.py once
per config. Designed around three constraints:

1. Shared free-tier quotas (see DAILY_REQUEST_LIMITS in config.py) -- a full
   cross product of LLM x embedding x chunk x k x 20 questions would blow the
   Claude budget in one afternoon. So we do one-factor-at-a-time from the
   verified-best base config (claude / openai emb / 2500/500 / k6) instead of
   a full cross product.
2. Index builds are the slow, quota-hungry part. Runs are sorted by index
   cache key so each embedding x chunk combination is built exactly once and
   every LLM sweep reuses it.
3. Reruns after a quota failure must be cheap: a run is skipped if its output
   CSV already exists and contains no error rows (override with --force).

Usage (from project root, venv active, keys in .env):
    python app/run_matrix.py --dry-run                 # print the plan + budget
    python app/run_matrix.py --stages llm              # just the LLM sweep
    python app/run_matrix.py --questions data/golden_set.csv --stages llm,chunk,k,emb
    python app/run_matrix.py --stages boundary,robust  # Stage-3 + paraphrase sets
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAGConfig, EMBEDDING_MODELS, provider_of  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
HARNESS = Path(__file__).resolve().parent / "eval_harness.py"

# Verified-best base config: every stage varies ONE axis off this.
BASE = dict(llm="claude-opus-4-8", emb="openai", cs=2500, co=500, k=6)

# chunk_size -> conventional overlap
CHUNK_PAIRS = {500: 50, 1500: 300, 2500: 500}


def make_plan(questions: str, stages: list) -> list:
    """Returns a list of run dicts: {name, llm, emb, cs, co, k, questions}."""
    runs = []

    def add(name, **kw):
        r = dict(BASE, questions=questions, name=name)
        r.update(kw)
        runs.append(r)

    if "llm" in stages:
        # Cloud + local models, all on the same (cheap-to-reuse) base index.
        for llm in ["claude-opus-4-8", "gpt-4o-mini", "gpt-4o",
                    "gemini-3.5-flash", "llama3.1", "mistral"]:
            add(f"llm_{llm}", llm=llm)

    if "chunk" in stages:
        for cs, co in CHUNK_PAIRS.items():
            if cs == BASE["cs"]:
                continue  # base config already covered by the llm stage
            add(f"chunk_{cs}", cs=cs, co=co)

    if "k" in stages:
        for k in [4, 8]:
            add(f"k{k}", k=k)

    if "emb" in stages:
        for emb in ["gemini", "ollama"]:
            add(f"emb_{emb}", emb=emb)

    if "boundary" in stages:
        # Boundary probing on the two most contrasting LLMs (honest vs eager).
        for llm in ["claude-opus-4-8", "gpt-4o-mini"]:
            add(f"boundary_{llm}", llm=llm,
                questions="data/boundary_set.csv")

    if "robust" in stages:
        for llm in ["claude-opus-4-8", "gpt-4o-mini"]:
            add(f"robust_{llm}", llm=llm,
                questions="data/robustness_set.csv")

    # De-duplicate identical configs (e.g. base appears in several stages).
    seen, unique = set(), []
    for r in runs:
        key = (r["llm"], r["emb"], r["cs"], r["co"], r["k"], r["questions"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Sort so identical index caches (embedding x chunk) run back-to-back.
    unique.sort(key=lambda r: (r["emb"], r["cs"], r["co"]))
    return unique


def out_path(run: dict) -> Path:
    qname = Path(run["questions"]).stem
    return RESULTS / f"mx_{run['name']}_{qname}.csv"


def already_done(path: Path) -> bool:
    """Done = file exists, has rows, and no row has a non-empty error."""
    if not path.exists():
        return False
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return bool(rows) and all(not (r.get("error") or "").strip() for r in rows)
    except Exception:
        return False


def count_questions(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def budget_report(runs: list):
    """LLM requests per provider = (# questions) per run. Index builds hit the
    embedding quota separately and are excluded here (hard to predict)."""
    per_provider = {}
    for r in runs:
        qfile = PROJECT_ROOT / r["questions"]
        n = count_questions(qfile) if qfile.exists() else 0
        prov = provider_of(r["llm"])
        per_provider[prov] = per_provider.get(prov, 0) + n
    print("\nEstimated LLM requests this session (vs. daily limit in config.py):")
    from config import DAILY_REQUEST_LIMITS
    for prov, n in sorted(per_provider.items()):
        limit = DAILY_REQUEST_LIMITS.get(prov, "∞")
        flag = "  ⚠️ OVER BUDGET" if isinstance(limit, int) and n > limit else ""
        print(f"  {prov:8s} {n:4d} / {limit}{flag}")
    print("  (index builds consume embedding quota on top of this)\n")


def main():
    p = argparse.ArgumentParser(description="Run the staged ablation matrix.")
    p.add_argument("--questions", default="data/smoke_test.csv",
                   help="Default question set for llm/chunk/k/emb stages")
    p.add_argument("--stages", default="llm,chunk,k,emb",
                   help="Comma list from: llm,chunk,k,emb,boundary,robust")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Rerun even if a clean output CSV already exists")
    args = p.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    runs = make_plan(args.questions, stages)

    print(f"Planned runs ({len(runs)}):")
    for r in runs:
        status = "SKIP (done)" if not args.force and already_done(out_path(r)) else "RUN"
        print(f"  [{status:11s}] {r['name']:24s} llm={r['llm']:18s} "
              f"emb={r['emb']:7s} cs={r['cs']}/{r['co']} k={r['k']} "
              f"q={r['questions']}")
    budget_report([r for r in runs if args.force or not already_done(out_path(r))])

    if args.dry_run:
        return

    failures = []
    for r in runs:
        out = out_path(r)
        if not args.force and already_done(out):
            continue
        cmd = [
            sys.executable, str(HARNESS),
            "--questions", r["questions"],
            "--llm", r["llm"],
            "--embedding-provider", r["emb"],
            "--chunk-size", str(r["cs"]),
            "--chunk-overlap", str(r["co"]),
            "--k", str(r["k"]),
            "--out", str(out),
        ]
        print(f"\n>>> {r['name']}: {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
        if rc != 0:
            failures.append(r["name"])
            print(f">>> {r['name']} exited {rc}; continuing with next run")

    print("\n==== matrix complete ====")
    if failures:
        print("Failed runs (rerun later, quotas reset daily):", ", ".join(failures))
    print("Next: python app/summarize_results.py")


if __name__ == "__main__":
    main()
