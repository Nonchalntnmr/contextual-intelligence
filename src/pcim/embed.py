"""Text embeddings.

The default `HashingEmbedder` is lexical, not semantic: signed feature hashing of word
unigrams, bigrams and character trigrams. It is deterministic, offline, and needs no
model download, which makes it the right default for tests and for a first run. It is
the wrong default for any quality claim — "car" and "automobile" are unrelated to it.
G1 (docs/04) requires a real embedding model, and the `Embedder` protocol is the seam.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim), rows L2-normalised
        ...


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                idx, sign = h % self.dim, 1.0 if (h >> 63) & 1 else -1.0
                weight = 0.5 if f.startswith("c:") else 1.0
                counts[idx] = counts.get(idx, 0.0) + sign * weight
            for idx, v in counts.items():
                out[row, idx] = np.sign(v) * np.log1p(abs(v))
            n = np.linalg.norm(out[row])
            if n > 0:
                out[row] /= n
        return out
