"""Response formatting utilities"""

from typing import List, Dict, Any


def format_sources(source_nodes: List[Any]) -> str:
    """Format source citations. Works with any document metadata."""
    if not source_nodes:
        return "No sources available"

    sources = []
    for i, node in enumerate(source_nodes, 1):
        meta = getattr(node, "metadata", {}) or {}
        source = f"Source {i}: {meta.get('source', 'Unknown')}"
        # Append any optional enrichment fields present in metadata
        extras = []
        if "company_ticker" in meta:
            extras.append(f"Ticker: {meta['company_ticker']}")
        if "fiscal_year" in meta:
            extras.append(f"Year: {meta['fiscal_year']}")
        if extras:
            source += " (" + " | ".join(extras) + ")"
        sources.append(source)

    return "\n".join(sources)


def calculate_confidence(score: float, source_count: int) -> float:
    """Calculate confidence score"""
    if source_count == 0:
        return 0.0
    
    base_confidence = min(score * 100, 95.0)
    
    source_boost = min(source_count * 5, 15)
    
    return min(base_confidence + source_boost, 99.0)


def nodes_to_source_list(source_nodes: List[Any]) -> List[Dict[str, Any]]:
    """Convert retrieved nodes to list of dicts for UI (file_name, score, text)."""
    out = []
    for node in source_nodes or []:
        meta = getattr(node, "metadata", {}) or {}
        score = getattr(node, "score", 0.0) or 0.0
        text = node.get_content() if hasattr(node, "get_content") else str(node)
        file_name = meta.get("source", meta.get("file_path", meta.get("file_name", "Unknown")))
        if isinstance(file_name, (list, tuple)):
            file_name = file_name[0] if file_name else "Unknown"
        out.append({"file_name": str(file_name), "score": float(score), "text": text})
    return out


def format_query_response(
    answer: str,
    sources: str,
    confidence: float,
    blocked: bool = False,
    warning: str = None,
    source_nodes: List[Any] = None,
) -> Dict[str, Any]:
    """Format final response."""
    conf = round(confidence, 1)
    response = {
        "answer": answer,
        "sources": sources,
        "confidence": conf,
        "metadata": {
            "blocked": blocked,
            "confidence": conf,
        },
    }
    if warning:
        response["metadata"]["warning"] = warning
    if source_nodes is not None:
        response["source_nodes"] = nodes_to_source_list(source_nodes)
    return response
