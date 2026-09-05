"""Example usage that touches ONLY the bundle tree.

Run from the bundle root with PYTHONPATH pointing at this ``student/`` directory:

    PYTHONPATH=student python student/example.py

The script scrubs course repo paths from ``sys.path``, refuses to import usim, coms,
or mplc, then loads the catalogue, embeds a query, and ranks items with BM25.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_STUDENT_DIR = Path(__file__).resolve().parent
_COURSE_MARKERS = ("rss-usim/src", "rss-coms/src", "rss-mplc/src")


def _scrub_sys_path() -> None:
    """Keep numpy/pyarrow, drop editable course src roots."""
    cleaned: list[str] = []
    for entry in sys.path:
        text = str(entry).replace("\\", "/")
        if any(marker in text for marker in _COURSE_MARKERS):
            continue
        if text.rstrip("/").endswith(("rss-usim/src", "rss-coms/src", "rss-mplc/src")):
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned
    student = str(_STUDENT_DIR)
    if student in sys.path:
        sys.path.remove(student)
    sys.path.insert(0, student)
    for name in list(sys.modules):
        root = name.split(".", 1)[0]
        if root in ("usim", "coms", "mplc"):
            del sys.modules[name]


def _refuse_course_imports() -> None:
    for name in ("usim", "coms", "mplc"):
        if name in sys.modules:
            raise SystemExit(f"refusing to run with '{name}' already imported")
        try:
            __import__(name)
        except ImportError:
            continue
        raise SystemExit(f"refusing to run: '{name}' is importable on sys.path")


def main() -> int:
    _scrub_sys_path()
    _refuse_course_imports()

    # This file lives at bundle_v1/student/example.py; the parquet files sit one level up.
    bundle_dir = _STUDENT_DIR.parent
    if not (bundle_dir / "catalogue.parquet").is_file():
        raise SystemExit(f"catalogue.parquet is missing under {bundle_dir}")

    from bm25 import Bm25Index
    from embedder import EmbedderConfig, MockHashEmbedder
    from load import item_texts, load_catalogue, load_queries, load_schemas

    schemas = load_schemas(bundle_dir)
    catalogue = load_catalogue(bundle_dir)
    queries = load_queries(bundle_dir)
    texts = item_texts(catalogue)
    bm25 = Bm25Index(texts, k1=float(schemas["bm25"]["k1"]),
                     b=float(schemas["bm25"]["b"]))
    embedder = MockHashEmbedder(EmbedderConfig.from_schemas(schemas))

    query_text = str(queries["text"][0])
    scores = bm25.scores(query_text)
    top = int(scores.argmax()) if scores.size else -1
    vector = embedder.encode([query_text])[0]
    norm = float((vector ** 2).sum() ** 0.5)
    # Cosine against published item embeddings: the semantic half of a student baseline.
    item_vectors = np.asarray(catalogue["embedding"], dtype=np.float64)
    item_norm = (item_vectors ** 2).sum(axis=1) ** 0.5
    item_norm = np.where(item_norm > 0.0, item_norm, 1.0)
    cosine = (item_vectors @ vector) / (item_norm * (norm if norm > 0.0 else 1.0))
    sem_top = int(cosine.argmax()) if cosine.size else -1

    print(f"world_id={schemas['world_id']}")
    print(f"items={len(texts)} queries={len(queries['query_id'])}")
    print(f"query={query_text!r}")
    print(f"bm25_top_item={catalogue['item_id'][top] if top >= 0 else None}")
    print(f"sem_top_item={catalogue['item_id'][sem_top] if sem_top >= 0 else None}")
    print(f"embedding_dim={int(vector.shape[0])} embedding_norm={norm:.4f}")
    if "exploration" in schemas:
        print(f"exploration_bootstrap_impressions="
              f"{schemas['exploration'].get('bootstrap_impressions_per_query')}")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
