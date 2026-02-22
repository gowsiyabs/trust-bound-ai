"""
Embeddings manager.

Supports two providers selectable via EMBEDDING_PROVIDER env var:
  openai        → text-embedding-3-small (default, recommended for cost)
  huggingface   → BAAI/bge-large-en-v1.5 (local, free, requires GPU/CPU)

Cost guide (as of 2025):
  text-embedding-3-small  $0.020 / 1M tokens  ← recommended
  text-embedding-3-large  $0.130 / 1M tokens
  BAAI/bge-large-en-v1.5  free (local compute cost only)

Switch with one env-var in .env:
  EMBEDDING_PROVIDER=openai
  OPENAI_API_KEY=sk-...
"""

import os
from llama_index.core import Settings

from ..utils.config import EmbeddingConfig, get_global_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingModelManager:
    """Manages embedding model – supports OpenAI and HuggingFace providers."""

    def __init__(self, config: EmbeddingConfig = None):
        if config is None:
            config = get_global_config().rag.embeddings

        self.config = config
        self.embed_model = self._load_model()
        self._set_global_model()
        logger.info(
            f"Embeddings ready: provider={config.provider}  "
            f"model={self._model_label()}"
        )

    def _model_label(self) -> str:
        if self.config.provider == "openai":
            return self.config.openai_model
        return self.config.model_name

    def _load_model(self):
        if self.config.provider == "openai":
            return self._load_openai()
        return self._load_huggingface()

    def _load_openai(self):
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError:
            raise ImportError(
                "llama-index-embeddings-openai is required for OpenAI embeddings.\n"
                "Install it with:  pip install llama-index-embeddings-openai"
            )

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file:\n"
                "  OPENAI_API_KEY=sk-..."
            )

        kwargs = {
            "model": self.config.openai_model,
            "api_key": api_key,
        }
        if self.config.openai_dimensions:
            kwargs["dimensions"] = self.config.openai_dimensions

        return OpenAIEmbedding(**kwargs)

    def _load_huggingface(self):
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError:
            raise ImportError(
                "llama-index-embeddings-huggingface is required for local embeddings.\n"
                "Install it with:  pip install llama-index-embeddings-huggingface"
            )

        from .device_selector import get_optimal_device
        device = get_optimal_device()

        return HuggingFaceEmbedding(
            model_name=self.config.model_name,
            device=device,
            embed_batch_size=self.config.batch_size,
        )

    def _set_global_model(self):
        Settings.embed_model = self.embed_model

    def get_model(self):
        return self.embed_model


def initialize_embeddings(config: EmbeddingConfig = None) -> EmbeddingModelManager:
    """Initialize and return the embedding manager."""
    return EmbeddingModelManager(config)


# ── EmbeddingManager alias for backwards compatibility ────────────────────────
EmbeddingManager = EmbeddingModelManager
