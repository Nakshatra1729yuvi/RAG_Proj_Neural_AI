import re
from typing import List, Dict, Any, Optional

def calculate_hit_rate(retrieved_docs: List[str], expected_doc: str) -> float:
    """HitRate: 1.0 if expected_doc is anywhere in retrieved_docs, else 0.0"""
    if expected_doc == "none":
        return 1.0 if not retrieved_docs else 0.0
    norm_expected = expected_doc.lower()
    for d in retrieved_docs:
        if norm_expected in d.lower() or d.lower() in norm_expected:
            return 1.0
    return 0.0

def calculate_reciprocal_rank(retrieved_docs: List[str], expected_doc: str) -> float:
    """MRR contribution: 1/rank of first occurrence, else 0.0"""
    if expected_doc == "none":
        return 1.0 if not retrieved_docs else 0.0
    norm_expected = expected_doc.lower()
    for idx, d in enumerate(retrieved_docs, start=1):
        if norm_expected in d.lower() or d.lower() in norm_expected:
            return 1.0 / idx
    return 0.0

def calculate_precision_at_k(retrieved_docs: List[str], expected_doc: str, k: int) -> float:
    """Precision@K: Proportion of top K docs matching expected_doc"""
    if not retrieved_docs or k <= 0:
        return 0.0
    if expected_doc == "none":
        return 1.0 if not retrieved_docs else 0.0
    top_k = retrieved_docs[:k]
    norm_expected = expected_doc.lower()
    hits = sum(1 for d in top_k if norm_expected in d.lower() or d.lower() in norm_expected)
    return hits / len(top_k)

def calculate_recall_at_k(retrieved_docs: List[str], expected_doc: str, k: int) -> float:
    """Recall@K: 1.0 if expected doc found in top K, else 0.0 (for single-target benchmark)"""
    return calculate_hit_rate(retrieved_docs[:k], expected_doc)

def calculate_lexical_faithfulness(generated_answer: str, ground_truth: str) -> float:
    """Computes token overlap / containment faithfulness score between 0.0 and 1.0"""
    gen_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', generated_answer.lower()))
    gt_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', ground_truth.lower()))
    if not gt_tokens:
        return 1.0
    overlap = len(gen_tokens.intersection(gt_tokens)) / len(gt_tokens)
    return min(1.0, round(float(overlap), 3))
