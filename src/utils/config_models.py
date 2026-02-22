"""Pydantic config models - part 1"""

from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent


class EmbeddingConfig(BaseModel):
    """
    Embedding model configuration.

    Provider selection (highest priority wins):
      1. Env var  EMBEDDING_PROVIDER  (openai | huggingface)
      2. Auto-detect: if OPENAI_API_KEY is set → openai
      3. YAML value → falls back to "huggingface"

    Recommended for cost efficiency:
      provider: openai
      openai_model: text-embedding-3-small   (~$0.02 / 1M tokens)
    """
    provider: Literal["huggingface", "openai"] = Field(default="huggingface")

    # HuggingFace (local, free)
    model_name: str = Field(default="BAAI/bge-large-en-v1.5")
    device: Literal["cuda", "cpu", "mps"] = Field(default="cpu")
    batch_size: int = Field(default=32, ge=1)

    # OpenAI
    openai_model: str = Field(default="text-embedding-3-small")
    openai_dimensions: Optional[int] = Field(
        default=None,
        description="Optional: reduce output dimensions (e.g. 512) to lower storage cost.",
    )

    def model_post_init(self, __context):
        env_provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        if env_provider in ("openai", "huggingface"):
            object.__setattr__(self, "provider", env_provider)
        elif not env_provider and openai_key:
            object.__setattr__(self, "provider", "openai")


class ChunkingConfig(BaseModel):
    """
    Document chunking configuration.

    OpenAI cost note:
      Smaller chunks = fewer tokens per embedding call = lower cost.
      512 tokens is a good balance for 10-K Q&A (one idea per chunk).
      Overlap of 100 preserves sentence continuity without doubling cost.
    """
    chunk_size: int = Field(default=512, ge=128, le=4096)
    chunk_overlap: int = Field(default=100, ge=0)
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v, info):
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class RetrievalConfig(BaseModel):
    """
    Retrieval configuration.

    OpenAI cost note:
      top_k controls how many chunks land in the generation prompt.
      5 chunks × 512 tokens = ~2,560 context tokens per query.
      Increasing to 10 roughly doubles generation cost with diminishing returns.
    """
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_hybrid_search: bool = Field(default=True)
    use_query_rewriting: bool = Field(default=True)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    """
    LLM configuration.

    Priority order (highest wins):
      1. Env var  LLM_PROVIDER  (ollama | openai)
      2. Env var  LLM_MODEL     (e.g. gpt-4o or llama3.2)
      3. YAML / programmatic values passed in
      4. These defaults below

    Auto-detect: if LLM_PROVIDER is not set but OPENAI_API_KEY is present
    in the environment, the provider is automatically switched to "openai".
    """
    provider: Literal["ollama", "openai"] = Field(default="ollama")
    model: str = Field(default="llama3.2")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(
        default=512, ge=1,
        description="gpt-4o-mini at 512 output tokens is fast and cheap. "
                    "Raise to 1024 only if answers are being cut off.",
    )
    timeout: float = Field(default=120.0)
    base_url: Optional[str] = Field(default="http://127.0.0.1:11434")
    api_key: Optional[str] = Field(default=None)

    def model_post_init(self, __context):
        # 1. Env-var overrides (allow switching without touching YAML)
        env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        env_model = os.getenv("LLM_MODEL", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        ollama_url = os.getenv("OLLAMA_BASE_URL", "").strip()

        # Apply provider override
        if env_provider in ("ollama", "openai"):
            object.__setattr__(self, "provider", env_provider)
        elif not env_provider and openai_key:
            # Auto-detect: API key present → use OpenAI
            object.__setattr__(self, "provider", "openai")

        # Apply model override
        if env_model:
            object.__setattr__(self, "model", env_model)
        elif self.provider == "openai" and self.model == "llama3.2":
            # Sensible default when switching to OpenAI without specifying model
            object.__setattr__(self, "model", "gpt-4o-mini")

        # Apply OpenAI key
        if self.provider == "openai":
            object.__setattr__(self, "api_key", openai_key or self.api_key)

        # Apply Ollama base_url override
        if self.provider == "ollama" and ollama_url:
            object.__setattr__(self, "base_url", ollama_url)


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    store_type: Literal["chroma", "qdrant"] = Field(default="chroma")
    persist_directory: Path = Field(default=PROJECT_ROOT / "data" / "processed" / "embeddings")
    collection_name: str = Field(default="rag_docs")
