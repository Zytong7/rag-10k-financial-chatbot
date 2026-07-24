# 10-K Financial Research Assistant — Engineering Setup

> Note: the GitHub repo URL (`rag-10k-financial-chatbot`) keeps its original
> name for link stability; the project itself is scoped and described below
> as a financial research assistant, not a general-purpose chatbot.

Financial research assistant that answers questions about Alphabet, Amazon,
and Microsoft's FY2025 10-K filings. Built on the course starter repo
([JHU-CDHAI/Chatbot-AIEB](https://github.com/JHU-CDHAI/Chatbot-AIEB)), with
every config the ablation experiments need (LLM, embedding, chunk size, k)
exposed as a switch, not a code edit.

## 1. Environment setup

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Local LLMs (optional, for the local-vs-cloud ablation):

```bash
brew install ollama              # or download from https://ollama.com/download
ollama pull llama3.1
ollama pull mistral
ollama pull nomic-embed-text     # embedding model for the Ollama path
```

## 2. API keys

Copy `.env.example` to `.env` and fill in your Gemini key
(get one at https://aistudio.google.com/apikey). `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY` are only needed if you select a Claude or GPT model,
respectively.

```bash
cp .env.example .env
# then edit .env and paste your key — never commit this file
```

`.env` is already in `.gitignore`. Do not put keys directly in code.

## 3. Run the assistant

```bash
streamlit run app/app.py
```

Sidebar switches:
- **LLM** — `gemini-3.5-flash`, `gemini-flash-latest`, `claude-opus-4-8`,
  `gpt-4o-mini`, `gpt-4o`, `llama3.1`, `mistral`
- **Embedding provider** — gemini, openai, or ollama (`nomic-embed-text`)
- **Chunk size / overlap**, **k per company**, **temperature**
- **API usage meters** — daily quota + RPM guard on shared free-tier keys

Each embedding+chunk combination gets its own cached index under
`.index_cache/` — changing config **never silently reuses a stale index**,
and switching only the LLM never triggers a rebuild.

## 4. Project layout

```
app/
  config.py       # all tunable settings + company/fiscal-year metadata
  models.py       # LLM / embedding factory (gemini / claude / ollama switch)
  prompts.py      # fiscal-year- and segment-aware system prompt
  ingest.py       # PDF loading, chunking, per-config FAISS index cache
  retrieval.py    # per-company balanced MMR retrieval + QA pipeline
  usage.py        # daily quota / RPM guard for shared API keys
  app.py          # Streamlit UI
  eval_harness.py # batch-run a golden set against one config -> CSV
data/
  10k/            # the 3 preloaded 10-K PDFs
  golden_set_template.csv  # starter template for the 15-20 question set
```

## 5. Design decisions worth knowing

- **Fiscal years differ.** Alphabet/Amazon FY2025 ends 2025-12-31; Microsoft's
  ends **2025-06-30**. The system prompt calls this out so the bot flags it on
  "end of 2025" questions instead of silently mixing periods.
- **Per-company balanced retrieval.** Comparative questions retrieve k chunks
  from *each* company's filing (metadata filter), so one company's documents
  can't crowd the others out of the context.
- **Chunk 1500/300** (vs 500/50 baseline) keeps most financial tables inside
  a single chunk.
- **Ollama `num_ctx` raised to 8192.** Ollama defaults to a 2048-token
  context and silently truncates longer prompts — without this fix,
  local-vs-cloud comparisons measure context truncation, not model quality.
- **Segment caveats in the prompt.** e.g. Microsoft's "Intelligent Cloud" ≠
  Azure; Amazon reports "Technology and infrastructure", not "R&D".

## 6. Running the golden set / ablation matrix

```bash
python app/eval_harness.py \
  --questions data/golden_set.csv \
  --llm gemini-3.5-flash \
  --embedding-provider gemini \
  --chunk-size 1500 --chunk-overlap 300 --k 4 \
  --out results/gemini_1500_k4.csv
```

Run once per config in the ablation matrix (swap `--llm llama3.1`, different
`--chunk-size`, etc.). Each run writes a CSV with the question, answer,
which companies were searched, page-level sources, and latency — stack the
CSVs to build the config x question pass/fail table.
