"""
Aggregates every results/*.csv into:

1. results/all_runs_long.csv  -- one row per (config, question) with an empty
   `grade` column. Graders fill grade with: correct / partial / wrong /
   honest_refusal / hallucination. This file is the single source of truth;
   re-running the script preserves grades already entered.
2. results/summary_matrix.md  -- config x question status matrix plus
   per-config counts, ready to paste into the tech note / slides.

Auto-status per answer (heuristic, NOT a grade -- grading stays manual):
    ERR      error column non-empty
    EMPTY    no answer text
    REFUSED  answer contains a refusal/insufficient-context marker
    ANS      an answer was produced (grader decides if it's right)

Stdlib only, so it runs anywhere without the RAG dependencies:
    python app/summarize_results.py
"""
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
LONG_CSV = RESULTS / "all_runs_long.csv"
SUMMARY_MD = RESULTS / "summary_matrix.md"

REFUSAL_MARKERS = [
    "don't have enough information",
    "do not have enough information",
    "not in the retrieved",
    "missing from the excerpts",
    "cannot compute",
    "cannot answer",
    "not present in the retrieved context",
    "无法回答",
    "上下文缺",
    "没有足够",
]

GRADE_VALUES = {"", "correct", "partial", "wrong", "honest_refusal", "hallucination"}


def config_label(row: dict) -> str:
    emb = row.get("embedding_provider", "?")
    return (f"{row.get('llm', '?')}|{emb}"
            f"|cs{row.get('chunk_size', '?')}/{row.get('chunk_overlap', '?')}"
            f"|k{row.get('k_per_company', '?')}")


def auto_status(row: dict) -> str:
    if (row.get("error") or "").strip():
        return "ERR"
    ans = (row.get("answer") or "").strip()
    if not ans:
        return "EMPTY"
    low = ans.lower()
    if any(m in low for m in REFUSAL_MARKERS):
        return "REFUSED"
    return "ANS"


def load_existing_grades() -> dict:
    """Keyed by (source_file, config, question id/conv+turn) so regenerating
    the long CSV never wipes manual grading work."""
    grades = {}
    if not LONG_CSV.exists():
        return grades
    with open(LONG_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r.get("source_file", ""), r.get("config", ""), r.get("qkey", ""))
            g = (r.get("grade") or "").strip().lower()
            if g:
                grades[key] = g
    return grades


def main():
    result_files = sorted(
        p for p in RESULTS.glob("*.csv")
        if p.name not in {LONG_CSV.name}
    )
    if not result_files:
        print(f"No result CSVs found in {RESULTS}")
        return

    existing = load_existing_grades()
    long_rows = []
    bad_grades = 0

    for path in result_files:
        with open(path, newline="", encoding="utf-8") as f:
            try:
                rows = list(csv.DictReader(f))
            except Exception as e:
                print(f"  skipping unreadable {path.name}: {e}")
                continue
        for r in rows:
            if "question" not in r:
                continue  # not a harness output (e.g. someone's scratch csv)
            qkey = r.get("id") or f"{r.get('conv_id', '?')}-t{r.get('turn', '?')}"
            cfg = config_label(r)
            key = (path.name, cfg, qkey)
            grade = existing.get(key, "")
            if grade not in GRADE_VALUES:
                bad_grades += 1
                grade = ""
            long_rows.append({
                "source_file": path.name,
                "config": cfg,
                "qkey": qkey,
                "question": (r.get("question") or "")[:160],
                "status": auto_status(r),
                "grade": grade,
                "expected_answer": (r.get("expected_answer") or "")[:160],
                "answer_first_200": re.sub(r"\s+", " ", r.get("answer") or "")[:200],
                "error_first_120": (r.get("error") or "")[:120],
                "latency_sec": r.get("latency_sec", ""),
            })

    LONG_CSV.parent.mkdir(exist_ok=True)
    with open(LONG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)

    # ---- matrix + per-config tallies -------------------------------------
    configs = sorted({r["config"] for r in long_rows})
    qkeys = sorted({r["qkey"] for r in long_rows})
    cell = {}
    for r in long_rows:
        # A grade, once entered, overrides the auto status in the matrix.
        display = r["grade"].upper() if r["grade"] else r["status"]
        cell[(r["config"], r["qkey"])] = display

    lines = ["# Ablation summary", "",
             f"Generated from {len(result_files)} result files, "
             f"{len(long_rows)} (config, question) rows.",
             "",
             "Cells show manual grade when entered (CORRECT/PARTIAL/WRONG/"
             "HONEST_REFUSAL/HALLUCINATION), otherwise auto status "
             "(ANS/REFUSED/ERR/EMPTY). Grade in results/all_runs_long.csv.",
             "", "## Config × question matrix", "",
             "| config | " + " | ".join(qkeys) + " |",
             "|" + "---|" * (len(qkeys) + 1)]
    for cfg in configs:
        row = [cell.get((cfg, q), "·") for q in qkeys]
        lines.append(f"| {cfg} | " + " | ".join(row) + " |")

    lines += ["", "## Per-config tallies", "",
              "| config | rows | answered | refused | errors | graded correct |",
              "|---|---|---|---|---|---|"]
    for cfg in configs:
        rows = [r for r in long_rows if r["config"] == cfg]
        n = len(rows)
        ans = sum(1 for r in rows if r["status"] == "ANS")
        ref = sum(1 for r in rows if r["status"] == "REFUSED")
        err = sum(1 for r in rows if r["status"] in ("ERR", "EMPTY"))
        cor = sum(1 for r in rows if r["grade"] == "correct")
        lines.append(f"| {cfg} | {n} | {ans} | {ref} | {err} | {cor} |")

    ungraded = sum(1 for r in long_rows if not r["grade"] and r["status"] == "ANS")
    lines += ["", f"Ungraded answered rows: **{ungraded}** "
              "(fill the grade column in all_runs_long.csv, then rerun this script)."]

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {LONG_CSV}  ({len(long_rows)} rows, {ungraded} awaiting grades)")
    if bad_grades:
        print(f"  note: {bad_grades} grade cells had unrecognized values and were cleared")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
