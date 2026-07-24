"""
Main Streamlit app. 10-Ks are preloaded (no upload step), and the sidebar
lets you flip every config the ablation matrix needs: LLM, embedding,
chunk size/overlap, k per company. Each embedding+chunk combination gets its
own cached index under .index_cache/ — switching config never silently
reuses a stale index, and switching only the LLM never rebuilds.

Run from the project root:
    streamlit run app/app.py
"""
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import (
    ALL_LLM_MODELS,
    COMPANIES,
    DAILY_REQUEST_LIMITS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    GEMINI_EMBEDDING_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    SESSION_MAX_ROUNDS,
    RAGConfig,
)
from ingest import build_vector_store
from retrieval import answer_question, detect_companies
from usage import usage_today

st.set_page_config(page_title="10-K RAG Chatbot", page_icon="📊", layout="wide")
st.title("📊 Alphabet / Amazon / Microsoft 10-K Chatbot")
st.caption(
    "Ask about the FY2025 10-K filings. Alphabet & Amazon cover Jan 1 – Dec 31, 2025; "
    "Microsoft's fiscal year runs Jul 1, 2024 – Jun 30, 2025."
)

with st.sidebar:
    st.header("⚙️ Configuration")

    llm_model = st.selectbox("LLM", ALL_LLM_MODELS, index=0)

    from config import EMBEDDING_MODELS
    embedding_provider = st.selectbox("Embedding provider", list(EMBEDDING_MODELS), index=0)
    embedding_model = EMBEDDING_MODELS[embedding_provider]
    st.caption(f"Embedding model: `{embedding_model}`")

    chunk_size = st.slider("Chunk size", 500, 2000, DEFAULT_CHUNK_SIZE, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, DEFAULT_CHUNK_OVERLAP, step=50)
    k_per_company = st.slider("Chunks per company (k)", 2, 8, 4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)

    cfg = RAGConfig(
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        temperature=temperature,
        k_per_company=k_per_company,
    )

    st.divider()
    st.header("📈 API usage today")
    for prov, limit in DAILY_REQUEST_LIMITS.items():
        used = usage_today(prov)
        st.progress(min(used / limit, 1.0), text=f"{prov}: {used}/{limit}")
    st.caption("Ollama (local) is unlimited.")

    st.divider()
    st.caption("Config fingerprint (for the ablation matrix):")
    st.code(
        f"{cfg.llm_model} | {cfg.embedding_provider} | "
        f"chunk={cfg.chunk_size}/{cfg.chunk_overlap} | k={cfg.k_per_company}",
        language=None,
    )

    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource(show_spinner="Building/loading the vector index for this config...")
def cached_index(index_key: str, _cfg: RAGConfig):
    return build_vector_store(_cfg)


try:
    vector_store = cached_index(cfg.index_key(), cfg)
except Exception as e:
    st.error(
        f"Could not build/load the index: {e}\n\n"
        "Check your .env keys (gemini) or that Ollama is running (ollama embeddings)."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

rounds_used = len(st.session_state.messages) // 2
if rounds_used >= SESSION_MAX_ROUNDS:
    st.warning(
        f"Session limit reached ({SESSION_MAX_ROUNDS} rounds). "
        "Click 'Clear conversation' to start fresh."
    )
    st.stop()

user_input = st.chat_input(
    f"Ask about the 10-K filings...  (round {rounds_used + 1}/{SESSION_MAX_ROUNDS})"
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner(f"Thinking with {cfg.llm_model}..."):
        try:
            result = answer_question(
                vector_store, cfg, user_input, st.session_state.messages[:-1]
            )
        except RuntimeError as e:
            st.error(str(e))
            st.session_state.messages.pop()
            st.stop()
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.session_state.messages.pop()
            st.stop()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in result.text.split():
            full_response += chunk + " "
            placeholder.markdown(full_response)
            time.sleep(0.02)

        with st.expander(f"View {len(result.sources)} retrieved chunks (context)"):
            st.caption(
                f"Searched: {', '.join(result.companies_searched)} · "
                f"k={cfg.k_per_company} per company"
            )
            for i, doc in enumerate(result.sources):
                st.markdown(
                    f"**Chunk {i + 1}** — {doc.metadata.get('company')} 10-K, "
                    f"page {doc.metadata.get('page', '?')} "
                    f"(FY end {doc.metadata.get('fiscal_year_end')})"
                )
                st.text(doc.page_content[:600])
                st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": result.text})
