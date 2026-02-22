"""Score fusion utilities for hybrid retrieval"""

from typing import List, Dict, Any
import math


def reciprocal_rank_fusion(results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Fuse multiple ranked lists using RRF"""
    fused_scores = {}
    
    for results in results_list:
        for rank, result in enumerate(results, 1):
            doc_id = result.get("id", id(result))
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    "score": 0.0,
                    "result": result
                }
            
            fused_scores[doc_id]["score"] += 1.0 / (k + rank)
    
    sorted_results = sorted(
        fused_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    
    return [item["result"] for item in sorted_results]


def normalize_scores(nodes: List[Any]) -> List[Any]:
    """Normalize retrieval scores"""
    if not nodes:
        return nodes
    
    scores = [node.score for node in nodes if hasattr(node, 'score')]
    
    if not scores:
        return nodes
    
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    if score_range == 0:
        return nodes
    
    for node in nodes:
        if hasattr(node, 'score'):
            node.score = (node.score - min_score) / score_range
    
    return nodes
