import re
import json
from pathlib import Path
from typing import List, Tuple, Optional
from rank_bm25 import BM25Plus
from backend.ingestion.chunker import DocumentChunk

class BM25Index:
    """
    BM25 retriever optimized for biomedical and healthcare terminology using BM25+.
    Prevents negative/zero IDF pathology on small subsets and preserves hyphens in medical compound words.
    """

    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do",
        "does", "doing", "down", "during", "each", "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on",
        "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
        "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
        "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
        "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
        "with", "would", "you", "your", "yours", "yourself", "yourselves"
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Plus] = None

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        # Keep letters, numbers, and hyphens within words (e.g. type-2, COVID-19)
        tokens = re.findall(r'[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*', text.lower())
        return [t for t in tokens if len(t) > 1 and t not in cls.STOP_WORDS]

    def build_index(self, chunks: List[DocumentChunk]):
        self.chunks = list(chunks)
        self.tokenized_corpus = [self.tokenize(c.text) for c in self.chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Plus(self.tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if self.bm25 is None or len(self.chunks) == 0:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Rank descending
        scored_indices = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:min(top_k, len(scored_indices))]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx, score in top_indices:
            # Check if at least one query term appears in this chunk
            chunk_tokens = set(self.tokenized_corpus[idx])
            has_term_overlap = any(q_term in chunk_tokens for q_term in tokenized_query)
            if has_term_overlap and score > 0:
                results.append((self.chunks[idx], float(score)))

        return results

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta,
            "chunks": [chunk.model_dump() for chunk in self.chunks]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self.delta = data.get("delta", 1.0)
            chunks = [DocumentChunk(**item) for item in data.get("chunks", [])]
            self.build_index(chunks)
            return True
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            self.clear()
            return False

    def clear(self):
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

    def count(self) -> int:
        return len(self.chunks)
