"""
Central config for the RAG chatbot. Every knob the ablation experiments need to
flip (LLM, embedding, chunk size/overlap, k) lives here or is overridable via
env vars, so switching a setting never means editing app logic.
"""
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "10k"))
INDEX_CACHE_DIR = PROJECT_ROOT / ".index_cache"
USAGE_FILE = PROJECT_ROOT / "usage.json"

# --- Companies & fiscal year facts -----------------------------------------
# Fiscal year end dates matter: Microsoft's FY doesn't line up with the
# calendar year the way Amazon's and Alphabet's do. See prompts.py for how
# this gets surfaced to the model.
COMPANIES = {
    "Alphabet": {
        "file": DATA_DIR / "Alphabet" / "Alphabet_10k_2025.pdf",
        "fiscal_year_end": "2025-12-31",
        "aliases": ["alphabet", "google", "googl", "goog", "gcp", "youtube"],
    },
    "Amazon": {
        "file": DATA_DIR / "Amazon" / "Amazon_10k_2025.pdf",
        "fiscal_year_end": "2025-12-31",
        "aliases": ["amazon", "aws", "amzn"],
    },
    "Microsoft": {
        "file": DATA_DIR / "Microsoft" / "Microsoft_10K_2025.pdf",
        "fiscal_year_end": "2025-06-30",
        "aliases": ["microsoft", "msft", "azure", "windows", "intelligent cloud"],
    },
}

# Words that signal a cross-company question even without naming companies.
COMPARATIVE_HINTS = ["compare", "each", "all three", "which company", "rank", "these companies", "versus", " vs "]

# --- Model choices -----------------------------------------------------------
# gemini-1.5-*/2.0-*/2.5-* are retired; 3.5-flash and flash-latest verified
# available (July 2026).
GEMINI_LLM_MODELS = ["gemini-3.5-flash", "gemini-flash-latest"]
CLAUDE_LLM_MODELS = ["claude-opus-4-8"]
OPENAI_LLM_MODELS = ["gpt-4o-mini", "gpt-4o"]
OLLAMA_LLM_MODELS = ["llama3.1", "mistral"]
ALL_LLM_MODELS = GEMINI_LLM_MODELS + CLAUDE_LLM_MODELS + OPENAI_LLM_MODELS + OLLAMA_LLM_MODELS

# The course baseline's embedding-001 was retired by Google; this is the
# current stable replacement (verified via ListModels, July 2026).
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODELS = {
    "gemini": GEMINI_EMBEDDING_MODEL,
    "openai": OPENAI_EMBEDDING_MODEL,
    "ollama": OLLAMA_EMBEDDING_MODEL,
}

# 2500/500 verified: at 1500/300 Amazon's segment table was cut mid-row and
# AWS revenue never reached the model; 500/50 (course baseline) is worse still.
DEFAULT_CHUNK_SIZE = 2500
DEFAULT_CHUNK_OVERLAP = 500

# --- Rate limits (shared free-tier API keys) ---------------------------------
DAILY_REQUEST_LIMITS = {"gemini": 500, "claude": 100, "openai": 200}
RPM_LIMITS = {"gemini": 10, "claude": 30, "openai": 30}

HISTORY_WINDOW_ROUNDS = 6   # rounds of chat history included in the prompt
SESSION_MAX_ROUNDS = 30     # hard cap per session so history can't blow up


def provider_of(llm_model: str) -> str:
    if llm_model.startswith("gemini"):
        return "gemini"
    if llm_model.startswith("claude"):
        return "claude"
    if llm_model.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "ollama"


@dataclass
class RAGConfig:
    llm_model: str = "gemini-3.5-flash"
    embedding_provider: str = "gemini"
    embedding_model: str = GEMINI_EMBEDDING_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    temperature: float = 0.0
    k_per_company: int = 6      # chunks retrieved per company in scope
    fetch_k: int = 20           # MMR candidate pool per company
    # Ollama serves 2048-token contexts by default, which silently truncates
    # our multi-company prompts; raise it so local-vs-cloud ablations compare
    # models, not context limits.
    ollama_num_ctx: int = 8192

    @property
    def llm_provider(self) -> str:
        return provider_of(self.llm_model)

    def index_key(self) -> str:
        """Identifies a vector store cache: only embedding + chunking affect it.
        LLM choice does NOT require rebuilding the index."""
        return f"{self.embedding_provider}_{self.embedding_model.replace('/', '-')}_cs{self.chunk_size}_co{self.chunk_overlap}"
