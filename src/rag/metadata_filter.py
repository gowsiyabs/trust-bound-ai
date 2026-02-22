"""Metadata filtering utilities"""

from typing import List, Dict, Any, Optional


def apply_metadata_filter(nodes: List[Any], filters: Optional[Dict[str, Any]]) -> List[Any]:
    """Filter nodes by metadata"""
    if not filters or not nodes:
        return nodes
    
    filtered = []
    
    for node in nodes:
        if _matches_filters(node.metadata, filters):
            filtered.append(node)
    
    # If filters are too strict (or metadata is missing), avoid dropping all
    # retrieval results; fallback to original nodes.
    return filtered if filtered else nodes


def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check if metadata matches filters"""
    for key, value in filters.items():
        if key not in metadata:
            return False
        
        if isinstance(value, list):
            if metadata[key] not in value:
                return False
        elif metadata[key] != value:
            return False
    
    return True


def extract_metadata_from_query(query: str) -> Dict[str, Any]:
    """Extract metadata filters from query"""
    filters = {}
    query_lower = query.lower()
    
    year_keywords = ["2023", "2024", "2025"]
    for year in year_keywords:
        if year in query:
            filters["fiscal_year"] = year
            break
    
    return filters
