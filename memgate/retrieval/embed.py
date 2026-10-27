"""Sentence embeddings, computed once at world generation and cached by text hash (plan D3-D5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from memgate.text import sha, tokenize

DIM = 384
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


class HashEmbedder:
    """Deterministic hashed bag-of-words. Offline fallback for tests; not for real experiments."""

    name = "hash"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(sha(tok, 8), 16)
                out[i, h % DIM] += 1.0 if (h >> 9) & 1 else -1.0
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
        return out


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(MINILM, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=len(texts) > 5000)
        return v.astype(np.float32)


def make_embedder(name: str):
    return MiniLMEmbedder() if name == "minilm" else HashEmbedder()


class EmbeddingStore:
    """text-hash -> unit vector. Misses are computed lazily with the same embedder and cached."""

    def __init__(self, embedder_name: str, keys: list[str] | None = None, mat: np.ndarray | None = None):
        self.embedder_name = embedder_name
        self._idx: dict[str, int] = {k: i for i, k in enumerate(keys or [])}
        self._mat = mat if mat is not None else np.zeros((0, DIM), dtype=np.float32)
        self._extra: dict[str, np.ndarray] = {}
        self._embedder = None
        self.misses = 0

    def add_texts(self, texts: list[str], embedder=None) -> None:
        new = sorted({t for t in texts if sha(t) not in self._idx})
        if not new:
            return
        emb = (embedder or self._get_embedder()).encode(new)
        start = len(self._idx)
        for j, t in enumerate(new):
            self._idx[sha(t)] = start + j
        self._mat = np.vstack([self._mat, emb]) if len(self._mat) else emb

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = make_embedder(self.embedder_name)
        return self._embedder

    def get(self, text: str) -> np.ndarray:
        k = sha(text)
        i = self._idx.get(k)
        if i is not None:
            return self._mat[i]
        v = self._extra.get(k)
        if v is None:
            self.misses += 1
            v = self._get_embedder().encode([text])[0]
            self._extra[k] = v
        return v

    def save(self, path: Path) -> None:
        keys = [None] * len(self._idx)
        for k, i in self._idx.items():
            keys[i] = k
        np.savez_compressed(path, keys=np.array(keys), mat=self._mat, embedder=np.array(self.embedder_name))

    @classmethod
    def load(cls, path: Path) -> EmbeddingStore:
        z = np.load(path, allow_pickle=False)
        return cls(str(z["embedder"]), [str(k) for k in z["keys"]], z["mat"].astype(np.float32))
