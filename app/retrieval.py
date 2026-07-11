"""
retrieval.py — Catalog loading and semantic search

Design:
- Embeddings: local sentence-transformers (all-MiniLM-L6-v2) when torch is
  available; HF Inference API as fallback for RAM-constrained deployments.
  Embeddings are cached to disk on first run — cold starts after that are instant.
- FAISS flat index (exact search) — correct choice for 377 items.
- Hybrid search: 70% semantic + 30% keyword BM25-style.
"""

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Embedding backend — local SentenceTransformer with HF API fallback
# ---------------------------------------------------------------------------
# Strategy: try local sentence-transformers first (fast, no network, exact).
# If torch/sentence-transformers isn't available (e.g. Render free-tier with
# tight RAM), fall back to the HuggingFace Inference API.

_local_model = None
_use_local: Optional[bool] = None  # None = not yet determined


def _try_load_local_model():
    global _local_model, _use_local
    if _use_local is not None:
        return _use_local
    try:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
        _use_local = True
        print("[Embed] Using local sentence-transformers model.")
    except Exception as e:
        _use_local = False
        print(f"[Embed] Local model unavailable ({e}), will use HF API.")
    return _use_local


HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def _get_embeddings(texts: list[str], retries: int = 3) -> np.ndarray:
    if _try_load_local_model():
        emb = _local_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return emb.astype(np.float32)

    # HF Inference API fallback
    import requests
    token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=60,
            )
            if response.status_code == 200:
                return _normalize(np.array(response.json(), dtype=np.float32))
            elif response.status_code == 503:
                wait = 2 ** attempt
                print(f"[HF API] Model loading, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"[HF API] Error {response.status_code}: {response.text[:200]}")
                time.sleep(1)
        except Exception as e:
            print(f"[HF API] Request failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)

    print("[Embed] All embedding backends failed — semantic search disabled.")
    return np.zeros((len(texts), 384), dtype=np.float32)


# ---------------------------------------------------------------------------
# Catalog item wrapper
# ---------------------------------------------------------------------------

class CatalogItem:
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
        parts = [
            self.name,
            self.description,
            " ".join(self.categories),
            " ".join(self.job_levels),
        ]
        return " | ".join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------
# FAISS index — built once at startup
# ---------------------------------------------------------------------------

class CatalogIndex:
    def __init__(self, catalog_path: str | Path):
        import faiss

        catalog_path = Path(catalog_path)
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.items: list[CatalogItem] = [CatalogItem(d) for d in raw]

        texts = [item.embed_text() for item in self.items]

        # Cache embeddings to disk — avoids re-hitting HF API on every cold start.
        # Cache key is a hash of the embed texts, so it auto-invalidates if catalog changes.
        import hashlib
        cache_key = hashlib.md5("".join(texts).encode()).hexdigest()[:12]
        cache_path = Path(os.environ.get("EMBED_CACHE_PATH", "data/embeddings_cache.npy"))
        cache_meta_path = cache_path.with_suffix(".meta")

        embeddings = None
        if cache_path.exists() and cache_meta_path.exists():
            if cache_meta_path.read_text().strip() == cache_key:
                print("Loading embeddings from cache...")
                embeddings = np.load(str(cache_path)).astype(np.float32)

        if embeddings is None:
            print(f"Building embeddings for {len(self.items)} catalog items via HF API...")
            batch_size = 64
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = _get_embeddings(batch)
                all_embeddings.append(batch_embeddings)
                if i + batch_size < len(texts):
                    time.sleep(0.5)
            embeddings = np.vstack(all_embeddings).astype(np.float32)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), embeddings)
            cache_meta_path.write_text(cache_key)
            print("Embeddings cached to disk.")

        # FAISS flat inner-product index (exact search, correct for <1000 items)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.by_slug: dict[str, CatalogItem] = {item.slug: item for item in self.items}
        self.by_url: dict[str, CatalogItem] = {item.url.rstrip("/"): item for item in self.items}
        print(f"Index ready: {len(self.items)} items loaded.")

    def semantic_search(self, query: str, top_k: int = 10) -> list[tuple[CatalogItem, float]]:
        q_vec = _get_embeddings([query]).astype(np.float32)
        scores, indices = self.index.search(q_vec, min(top_k, len(self.items)))
        return [(self.items[idx], float(score))
                for score, idx in zip(scores[0], indices[0]) if idx >= 0]

    def keyword_search(self, query: str, top_k: int = 10) -> list[tuple[CatalogItem, float]]:
        q_terms = set(re.sub(r"[^a-z0-9\s]", " ", query.lower()).split())
        if not q_terms:
            return []
        scored = []
        for item in self.items:
            haystack = re.sub(r"[^a-z0-9\s]", " ",
                              (item.name + " " + item.description).lower()).split()
            freq = {t: haystack.count(t) for t in q_terms if t in haystack}
            score = sum(math.log(1 + f) for f in freq.values())
            if score > 0:
                scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.7) -> list[CatalogItem]:
        sem = self.semantic_search(query, top_k * 2)
        kw = self.keyword_search(query, top_k * 2)

        def norm(results):
            if not results:
                return {}
            max_s = max(s for _, s in results) or 1.0
            return {item.slug: s / max_s for item, s in results}

        sem_n, kw_n = norm(sem), norm(kw)
        all_slugs = set(sem_n) | set(kw_n)
        combined = {s: alpha * sem_n.get(s, 0) + (1 - alpha) * kw_n.get(s, 0)
                    for s in all_slugs}
        ranked = sorted(combined, key=lambda s: combined[s], reverse=True)[:top_k]
        return [self.by_slug[s] for s in ranked]

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