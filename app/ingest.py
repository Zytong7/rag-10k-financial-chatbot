"""
Loads the three 10-Ks, tags every chunk with company + fiscal year metadata,
and builds (or loads a cached) FAISS index for a given RAGConfig.

Only embedding_provider/model + chunk_size/overlap affect the index contents,
so the cache key (RAGConfig.index_key) deliberately ignores the LLM choice —
switching LLMs never triggers a rebuild.
"""
import re
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import COMPANIES, INDEX_CACHE_DIR, RAGConfig
from models import get_embeddings


def load_company_documents(companies=None):
    """Load PDF pages for the given companies (default: all), tagging each
    page with company + fiscal_year_end metadata."""
    companies = companies or list(COMPANIES.keys())
    documents = []
    for name in companies:
        info = COMPANIES[name]
        path = Path(info["file"])
        if not path.exists():
            raise FileNotFoundError(f"10-K for {name} not found at {path}")
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        for page in pages:
            page.metadata["company"] = name
            page.metadata["fiscal_year_end"] = info["fiscal_year_end"]
            # PyPDF numbers pages from 0; cite them 1-indexed to match PDF viewers.
            page.metadata["page"] = page.metadata.get("page", 0) + 1
        documents.extend(pages)
    return documents


def build_vector_store(cfg: RAGConfig, companies=None, force_rebuild=False):
    cache_dir = INDEX_CACHE_DIR / cfg.index_key()
    embeddings = get_embeddings(cfg)

    if cache_dir.exists() and not force_rebuild:
        return FAISS.load_local(
            str(cache_dir), embeddings, allow_dangerous_deserialization=True
        )

    documents = load_company_documents(companies)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    # Gemini's free embedding tier caps requests and tokens per minute, so a
    # one-shot FAISS.from_documents 429s on ~1000 chunks. Feed it in paced
    # batches, wait out 429s, and checkpoint after every batch so an aborted
    # build resumes instead of re-burning quota.
    pause = 30.0 if cfg.embedding_provider == "gemini" else 0.0
    vector_store = _embed_in_batches(chunks, embeddings, cache_dir, pause=pause)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(cache_dir))
    _partial_paths(cache_dir)[1].unlink(missing_ok=True)
    return vector_store


def _partial_paths(cache_dir):
    partial_dir = Path(str(cache_dir) + ".partial")
    return partial_dir, partial_dir / "progress.txt"


def _embed_in_batches(chunks, embeddings, cache_dir, batch_size=25, pause=0.0,
                      max_retries=10):
    partial_dir, progress_file = _partial_paths(cache_dir)
    vector_store, done = None, 0
    if progress_file.exists():
        done = int(progress_file.read_text())
        vector_store = FAISS.load_local(
            str(partial_dir), embeddings, allow_dangerous_deserialization=True
        )
        print(f"  resuming from checkpoint: {done}/{len(chunks)} chunks already embedded")

    total = len(chunks)
    for start in range(done, total, batch_size):
        batch = chunks[start:start + batch_size]
        for attempt in range(max_retries):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                break
            except Exception as e:
                msg = str(e)
                if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
                    raise
                m = re.search(r"retry in (\d+(?:\.\d+)?)", msg, re.IGNORECASE)
                delay = float(m.group(1)) + 5 if m else 60.0
                print(f"  rate-limited; waiting {delay:.0f}s "
                      f"(batch {start // batch_size + 1}, attempt {attempt + 1})")
                time.sleep(delay)
        else:
            raise RuntimeError(f"Embedding still rate-limited after {max_retries} retries; "
                               "checkpoint saved — rerun to resume")
        completed = min(start + batch_size, total)
        partial_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(partial_dir))
        progress_file.write_text(str(completed))
        print(f"  embedded {completed}/{total} chunks")
        if pause and completed < total:
            time.sleep(pause)
    return vector_store
