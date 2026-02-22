"""
promptfoo retrieval-only provider.

Called by promptfoo to test the retrieval step in isolation.
promptfoo passes `query` as the prompt string; this script returns the
top-K retrieved chunks as a single string.

promptfoo expects a `call_api(prompt, options, context)` function.
"""

import sys
from pathlib import Path

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_global_config
from src.utils.logging import setup_logging
from src.models.rag_model import load_rag_from_index


# Singleton – loaded once per promptfoo worker process
_model = None


def _get_model():
    global _model
    if _model is None:
        setup_logging(log_level="WARNING")
        _model = load_rag_from_index()
    return _model


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """
    Retrieval-only provider.

    Returns a string listing each retrieved chunk with its source and score.
    promptfoo can then run `contains`, `contains-all`, or `context-recall`
    assertions on this output.
    """
    try:
        model = _get_model()
        nodes = model.retriever.retrieve(prompt)

        if not nodes:
            return {"output": "(no results retrieved)"}

        lines = []
        for i, node in enumerate(nodes, 1):
            meta = getattr(node, "metadata", {}) or {}
            source = meta.get("source", "unknown")
            score = getattr(node, "score", 0.0) or 0.0
            text = node.get_content() if hasattr(node, "get_content") else str(node)
            lines.append(f"[{i}] source={source} score={score:.3f}\n{text[:400]}")

        return {"output": "\n\n".join(lines)}

    except Exception as e:
        return {"error": str(e), "output": ""}
