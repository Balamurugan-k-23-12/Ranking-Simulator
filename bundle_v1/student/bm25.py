"""BM25 over item text. Robertson form, default k1=1.2 and b=0.75.

Vendored into the student bundle. No imports outside the standard library and numpy.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np


def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase word tokens. Punctuation splits."""
    tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


class Bm25Index:
    """BM25 over a list of documents.

    The score of one document sums over the distinct words of the query:

        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))
    """

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        if k1 < 0.0:
            raise ValueError(f"bm25 k1 must not be negative, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"bm25 b must be in [0, 1], got {b}")
        self.k1 = float(k1)
        self.b = float(b)
        documents = [tokenize(text) for text in texts]
        self.n_docs = len(documents)
        self.lengths = np.array([len(words) for words in documents], dtype=np.float64)
        self.avg_length = float(self.lengths.mean()) if self.n_docs else 0.0
        self._postings, self._idf = _build_postings(documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def idf(self, term: str) -> float:
        return float(self._idf.get(term, 0.0))

    def scores(self, query_text: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0 or self.avg_length <= 0.0:
            return out
        norm = 1.0 - self.b + self.b * self.lengths / self.avg_length
        for term in dict.fromkeys(tokenize(query_text)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            rows, counts = posting
            saturation = counts * (self.k1 + 1.0) / (counts + self.k1 * norm[rows])
            out[rows] += self._idf[term] * saturation
        return out

    def normalized(self, query_text: str) -> np.ndarray:
        scores = self.scores(query_text)
        top = float(scores.max()) if scores.size else 0.0
        if top <= 0.0:
            return np.zeros_like(scores)
        return scores / top

    @classmethod
    def from_pickle(cls, path: str | Path) -> "Bm25Index":
        """Load a persisted public BM25 index written by the instructor cache."""
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        index = cls.__new__(cls)
        index.k1 = float(payload["k1"])
        index.b = float(payload["b"])
        index.n_docs = int(payload["n_docs"])
        index.lengths = np.asarray(payload["lengths"], dtype=np.float64)
        index.avg_length = float(payload["avg_length"])
        index._postings = payload["postings"]
        index._idf = payload["idf"]
        return index


_Postings = dict[str, tuple[np.ndarray, np.ndarray]]


def _build_postings(documents: Sequence[Sequence[str]]
                    ) -> tuple[_Postings, dict[str, float]]:
    counts: dict[str, dict[int, int]] = {}
    for row, words in enumerate(documents):
        for word in words:
            counts.setdefault(word, {})
            counts[word][row] = counts[word].get(row, 0) + 1
    n_docs = len(documents)
    postings: _Postings = {}
    idf: dict[str, float] = {}
    for term, per_doc in counts.items():
        rows = np.fromiter(per_doc.keys(), dtype=np.int64, count=len(per_doc))
        values = np.fromiter(per_doc.values(), dtype=np.float64, count=len(per_doc))
        postings[term] = (rows, values)
        frequency = float(rows.size)
        idf[term] = float(np.log(1.0 + (n_docs - frequency + 0.5) / (frequency + 0.5)))
    return postings, idf
