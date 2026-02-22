"""
promptfoo end-to-end RAG provider.

Called by promptfoo for full RAG evaluation (retrieval + LLM generation).
Returns a structured response so promptfoo can extract both the answer and
the retrieved context for faithfulness / relevance metrics.

promptfoo expects a `call_api(prompt, options, context)` function.

Response shape:
{
    "output": {
        "answer":    str,          # the LLM answer
        "context":   str,          # retrieved chunks joined as one string
        "sources":   list[str],    # source filenames
        "blocked":   bool,
        "confidence": float
    }
}

Use `options.transform: "output.answer"` in promptfooconfig.yaml to extract
just the answer text for string/semantic assertions.
Use `contextTransform: "output.context"` for faithfulness metrics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_global_config
from src.utils.logging import setup_logging
from src.models.rag_model import load_rag_from_index
from src.safety.input_validator import InputValidator
from src.safety.output_filter import OutputFilter


# Singletons
_model = None
_validator = None
_out_filter = None


def _get_components(safety: bool = True):
    global _model, _validator, _out_filter
    if _model is None:
        setup_logging(log_level="WARNING")
        _model = load_rag_from_index()
    if safety and _validator is None:
        _validator = InputValidator()
        _out_filter = OutputFilter(llm=_model.llm)
    return _model, _validator if safety else None, _out_filter if safety else None


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """
    Full RAG provider for promptfoo.

    `options.config.safety` (bool, default True) – set to false in
    promptfooconfig.yaml to bypass safety for raw model testing.
    """
    cfg = (options or {}).get("config", {})
    safety = cfg.get("safety", True)

    try:
        model, validator, out_filter = _get_components(safety=safety)

        result = model.query(
            prompt,
            input_validator=validator,
            output_filter=out_filter,
        )

        answer = result.get("answer", "")
        meta = result.get("metadata", {})
        source_nodes = result.get("source_nodes", [])

        # Build a single context string for promptfoo's context-* metrics
        context_text = "\n\n".join(
            n.get("text", "") for n in source_nodes if n.get("text")
        )
        sources = [n.get("file_name", "unknown") for n in source_nodes]

        return {
            "output": {
                "answer": answer,
                "context": context_text,
                "sources": sources,
                "blocked": meta.get("blocked", False),
                "confidence": meta.get("confidence", 0.0),
            },
            # Mirror retrieval context into metadata so promptfoo context-based
            # assertions can read it via contextTransform: "context.metadata.context".
            "metadata": {
                "context": context_text,
                "sources": sources,
                "blocked": meta.get("blocked", False),
                "confidence": meta.get("confidence", 0.0),
            },
        }

    except Exception as e:
        return {
            "error": str(e),
            "output": {
                "answer": "",
                "context": "",
                "sources": [],
                "blocked": False,
                "confidence": 0.0,
            },
        }
