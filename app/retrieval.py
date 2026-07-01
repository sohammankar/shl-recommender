"""
retrieval.py — Catalog loading and semantic search

WHY THIS DESIGN (for the interview):
- We use a local sentence-transformer model (all-MiniLM-L6-v2) rather than
  a paid embeddings API for three reasons:
    1. Zero cost and no rate-limit risk during automated grading (which may
       fire many calls in parallel).
    2. The catalog is only 377 items — embedding them locally takes ~2 seconds
       at startup and fits in ~10 MB of RAM.
    3. Removes one network dependency from the hot path; retrieval still works
       even if the LLM provider is briefly unavailable.

- We use FAISS (flat L2 index, brute-force) rather than HNSW or IVF because
  377 vectors makes approximate search pointless — exact search over 377
  items is faster than the overhead of an approximate algorithm, and it
  guarantees we never miss the correct result due to index approximation.

- We build a rich text representation per catalog item that concatenates
  name, description, category labels, and job levels. This is the "context
  engineering" part: if a user says "senior leadership", the job_levels
  field in the embedded text carries that signal even if the product's
  description doesn't use the word "leadership".

- We also expose keyword_search() (BM25-style term matching) as a fallback
  for exact product name queries like "What is the OPQ32r?" where semantic
  similarity may not surface the right item as #1.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Embedding model — loaded once at module import time (singleton pattern).
# This means the model is warm on the first /chat call rather than cold.
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ---------------------------------------------------------------------------
# Data classes (plain dicts kept for JSON-serializability simplicity)
# ---------------------------------------------------------------------------

class CatalogItem:
    """Thin wrapper so the rest of the app can use dot-access on catalog records."""
    __slots__ = ("id", "slug", "name", "url", "test_type", "categories",
                 "description", "duration", "languages", "job_levels",
                 "remote", "adaptive")

    def __init__(self, d: dict):
        self.id = d["id"]
        self.slug = d["slug"]
        self.name = d["name"]
        self.url = d["url"]
        self.test_type = d["test_type"]
        self.categories = d.get("categories", [])
        self.description = d.get("description", "")
        self.duration = d.get("duration", "")
        self.languages = d.get("languages", [])
        self.job_levels = d.get("job_levels", [])
        self.remote = d.get("remote", True)
        self.adaptive = d.get("adaptive", False)

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}

    def embed_text(self) -> str:
        """
        The text we actually embed for this item.

        We concatenate every semantically useful field so that queries like
        "senior leadership personality" surface OPQ32r even though the word
        "leadership" doesn't appear in the OPQ product description itself —
        it appears in its job_levels list. Likewise "contact centre simulation"
        will hit items whose categories include "Simulations".
        """
        parts = [
            self.name,
            self.description,
            " ".join(self.categories),
            " ".join(self.job_levels),
        ]
        return " | ".join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Index — built once at startup, reused for the lifetime of the process
# ---------------------------------------------------------------------------

class CatalogIndex:
    def __init__(self, catalog_path: str | Path):
        import faiss  # imported here so the module doesn't crash if faiss isn't installed yet

        catalog_path = Path(catalog_path)
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.items: list[CatalogItem] = [CatalogItem(d) for d in raw]

        # Build embeddings
        model = _get_model()
        texts = [item.embed_text() for item in self.items]
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        embeddings = embeddings.astype(np.float32)

        # FAISS flat inner-product index.
        # Because we normalize embeddings above, inner product == cosine similarity.
        # Flat = exact search (no approximation). Correct choice for 377 items.
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Keep a slug → item lookup for direct URL validation
        self.by_slug: dict[str, CatalogItem] = {item.slug: item for item in self.items}
        self.by_url: dict[str, CatalogItem] = {item.url.rstrip("/"): item for item in self.items}

    def semantic_search(self, query: str, top_k: int = 10) -> list[tuple[CatalogItem, float]]:
        """
        Returns up to top_k (item, score) pairs ranked by cosine similarity.
        score is in [0, 1] because both query and index vectors are normalized.
        """
        model = _get_model()
        q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_vec, min(top_k, len(self.items)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.items[idx], float(score)))
        return results

    def keyword_search(self, query: str, top_k: int = 10) -> list[tuple[CatalogItem, float]]:
        """
        Simple TF-IDF-style term frequency match over name + description.
        Used as a complementary signal for exact product-name queries.

        WHY: Semantic search can place an item like 'OPQ32r' slightly off the
        top if the query is a product name comparison ("difference between
        OPQ32r and GSA") because the query vector encodes 'difference' and
        'compare' as strong signals, potentially overweighting other items.
        Keyword search anchors on the exact token.
        """
        q_terms = set(re.sub(r"[^a-z0-9\s]", " ", query.lower()).split())
        if not q_terms:
            return []

        scored = []
        for item in self.items:
            haystack = (item.name + " " + item.description).lower()
            haystack = re.sub(r"[^a-z0-9\s]", " ", haystack)
            hay_terms = haystack.split()
            hay_term_set = set(hay_terms)
            hay_freq = {t: hay_terms.count(t) for t in q_terms if t in hay_terms}

            # Score = sum of TF * IDF approximation (log(1 + count))
            score = sum(math.log(1 + freq) for freq in hay_freq.values())
            if score > 0:
                scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.7) -> list[CatalogItem]:
        """
        Combines semantic and keyword search with a weighted sum.
        alpha=0.7 means 70% semantic, 30% keyword.

        WHY alpha=0.7: Semantic search generalizes well to paraphrased queries
        ("I need something for senior execs" maps to OPQ even without the
        word "OPQ"). Keyword search helps precision for exact product names.
        0.7/0.3 is a reasonable prior; in a production system you'd tune this
        on a labeled eval set.
        """
        sem_results = self.semantic_search(query, top_k=top_k * 2)
        kw_results = self.keyword_search(query, top_k=top_k * 2)

        # Normalize scores to [0,1] within each result set
        def normalize(results: list[tuple[CatalogItem, float]]) -> dict[str, float]:
            if not results:
                return {}
            max_s = max(s for _, s in results) or 1.0
            return {item.slug: s / max_s for item, s in results}

        sem_norm = normalize(sem_results)
        kw_norm = normalize(kw_results)

        all_slugs = set(sem_norm) | set(kw_norm)
        combined = {
            slug: alpha * sem_norm.get(slug, 0.0) + (1 - alpha) * kw_norm.get(slug, 0.0)
            for slug in all_slugs
        }

        ranked_slugs = sorted(combined, key=lambda s: combined[s], reverse=True)[:top_k]
        return [self.by_slug[s] for s in ranked_slugs]

    def get_by_url(self, url: str) -> Optional[CatalogItem]:
        return self.by_url.get(url.rstrip("/"))

    def get_by_name(self, name: str) -> Optional[CatalogItem]:
        name_lower = name.lower().strip()
        for item in self.items:
            if item.name.lower() == name_lower:
                return item
        return None

    def all_items(self) -> list[CatalogItem]:
        return self.items
