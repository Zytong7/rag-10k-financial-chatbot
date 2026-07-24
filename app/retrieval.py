"""
The question-answering pipeline: figure out which company(ies) a question is
about, retrieve a balanced set of chunks per company (so one company's
documents can't crowd out another's on comparative questions), and call the
configured LLM with the fiscal-year-aware system prompt.
"""
from dataclasses import dataclass

from config import COMPANIES, COMPARATIVE_HINTS, HISTORY_WINDOW_ROUNDS, RAGConfig
from models import get_llm
from prompts import ANSWER_TEMPLATE, SYSTEM_PERSONA
from usage import guarded_call


@dataclass
class Answer:
    text: str
    sources: list
    companies_searched: list


def detect_companies(question: str) -> list:
    """Companies to retrieve for. Naming specific companies scopes retrieval
    to them; comparative phrasing or no mention at all searches all three."""
    q = question.lower()
    mentioned = [
        name
        for name, info in COMPANIES.items()
        if any(alias in q for alias in info["aliases"])
    ]
    comparative = any(hint in q for hint in COMPARATIVE_HINTS)
    if not mentioned or comparative:
        return list(COMPANIES)
    return mentioned


def retrieve(vector_store, cfg: RAGConfig, question: str):
    """Per-company MMR retrieval: k chunks from each company in scope, so a
    three-company question always sees evidence from all three filings.
    The retrieval query is the question only — mixing in chat history or the
    persona would pollute the embedding search."""
    companies = detect_companies(question)
    docs = []
    for company in companies:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": cfg.k_per_company,
                "fetch_k": cfg.fetch_k,
                "lambda_mult": 0.7,
                "filter": {"company": company},
            },
        )
        docs.extend(retriever.invoke(question))
    return docs, companies


def format_history(messages, window_rounds: int = HISTORY_WINDOW_ROUNDS) -> str:
    """Most recent N rounds only — unbounded history slows every request and
    eventually overflows local-model context windows."""
    if not messages or window_rounds <= 0:
        return "(no prior conversation)"
    recent = messages[-(window_rounds * 2):]
    return "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    ) or "(no prior conversation)"


def build_context(docs) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('company', '?')} 10-K, page {d.metadata.get('page', '?')}, "
        f"FY end {d.metadata.get('fiscal_year_end', '?')}]\n{d.page_content}"
        for d in docs
    )


def answer_question(
    vector_store, cfg: RAGConfig, question: str, messages=()
) -> Answer:
    docs, companies = retrieve(vector_store, cfg, question)

    prompt = ANSWER_TEMPLATE.format(
        persona=SYSTEM_PERSONA,
        context=build_context(docs),
        chat_history=format_history(list(messages)),
        user_input=question,
    )

    llm = get_llm(cfg)

    def _call():
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    text = guarded_call(cfg.llm_provider, _call)
    return Answer(text=text, sources=docs, companies_searched=companies)
