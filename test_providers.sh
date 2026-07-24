#!/bin/sh
# Smoke-test all three cloud providers on the same 3 questions.
# Prerequisite: keys filled in .env. Run from the project root:  sh test_providers.sh
set -e
mkdir -p results

for MODEL in gemini-3.5-flash claude-opus-4-8 gpt-4o-mini; do
    echo "=========== $MODEL ==========="
    .venv/bin/python app/eval_harness.py \
        --questions data/smoke_test.csv \
        --llm "$MODEL" \
        --embedding-provider openai \
        --out "results/smoke_$MODEL.csv" || echo ">>> $MODEL FAILED (check its key in .env)"
done

echo
echo "Done. Compare answers against the expected_answer column:"
echo "  results/smoke_gemini-3.5-flash.csv"
echo "  results/smoke_claude-opus-4-8.csv"
echo "  results/smoke_gpt-4o-mini.csv"
