"""
Model factory: turns a model name into an actual LangChain LLM / embeddings
object. This is the whole point of the config switch — the ablation matrix
(LLM x embedding x chunk size) just calls these with different arguments.
"""
import os

from config import RAGConfig


def _require_env(var: str):
    if not os.environ.get(var):
        raise RuntimeError(f"{var} not set. Add it to your .env file.")


def get_llm(cfg: RAGConfig):
    provider = cfg.llm_provider

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        _require_env("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(model=cfg.llm_model, temperature=cfg.temperature)

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic

        _require_env("ANTHROPIC_API_KEY")
        # Current Claude models reject the temperature parameter.
        return ChatAnthropic(model=cfg.llm_model, max_tokens=4096)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        _require_env("OPENAI_API_KEY")
        return ChatOpenAI(model=cfg.llm_model, temperature=cfg.temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # num_ctx: Ollama defaults to a 2048-token context, which silently
        # drops the tail of multi-company prompts. See RAGConfig.
        return ChatOllama(
            model=cfg.llm_model, temperature=cfg.temperature, num_ctx=cfg.ollama_num_ctx
        )

    raise ValueError(f"Unknown llm provider for model: {cfg.llm_model}")


def get_embeddings(cfg: RAGConfig):
    if cfg.embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _require_env("GOOGLE_API_KEY")
        return GoogleGenerativeAIEmbeddings(model=cfg.embedding_model)

    if cfg.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        _require_env("OPENAI_API_KEY")
        return OpenAIEmbeddings(model=cfg.embedding_model)

    if cfg.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=cfg.embedding_model)

    raise ValueError(f"Unknown embedding_provider: {cfg.embedding_provider}")
