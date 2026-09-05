"""Load parquet and JSON from a written bundle.

Uses only numpy, pyarrow, and the standard library. Point ``bundle_dir`` at the
``bundle_v1`` directory that holds the parquet files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="ascii"))


def load_schemas(bundle_dir: str | Path) -> dict:
    return load_json(Path(bundle_dir) / "schemas.json")


def load_parquet(path: str | Path) -> dict[str, list]:
    """Read one parquet file into a column-name -> pylist mapping."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    return {name: table.column(name).to_pylist() for name in table.column_names}


def load_catalogue(bundle_dir: str | Path, *, with_embeddings: bool = True) -> dict[str, list]:
    """Read catalogue.parquet. Skip the embedding column when ``with_embeddings`` is false."""
    import pyarrow.parquet as pq

    path = Path(bundle_dir) / "catalogue.parquet"
    if with_embeddings:
        return load_parquet(path)
    table = pq.read_table(str(path))
    names = [name for name in table.column_names if name != "embedding"]
    slim = table.select(names)
    return {name: slim.column(name).to_pylist() for name in slim.column_names}


def load_catalogue_embeddings(bundle_dir: str | Path) -> np.ndarray:
    """Return item embeddings. Memory-map the sidecar when the bundle ships one."""
    root = Path(bundle_dir)
    schemas = load_schemas(root)
    sidecar = root / str(schemas.get("catalogue_embeddings", "catalogue_embeddings.f32.npy"))
    if sidecar.is_file():
        return np.load(sidecar, mmap_mode="r")
    cat = load_catalogue(root, with_embeddings=True)
    return embedding_matrix(cat["embedding"])


def load_users(bundle_dir: str | Path, *, heldout: bool = False) -> dict[str, list]:
    name = "users_heldout.parquet" if heldout else "users.parquet"
    return load_parquet(Path(bundle_dir) / name)


def load_queries(bundle_dir: str | Path) -> dict[str, list]:
    return load_parquet(Path(bundle_dir) / "queries.parquet")


def load_bootstrap_log(bundle_dir: str | Path) -> dict[str, list]:
    return load_parquet(Path(bundle_dir) / "bootstrap_log.parquet")


def load_query_embeddings(bundle_dir: str | Path) -> dict[str, list]:
    return load_parquet(Path(bundle_dir) / "query_embeddings.parquet")


def load_bm25_index(bundle_dir: str | Path):
    """Load ``catalogue_bm25.pkl`` when the bundle ships a persisted public index."""
    from bm25 import Bm25Index

    root = Path(bundle_dir)
    schemas = load_schemas(root)
    name = schemas.get("bm25_index", "catalogue_bm25.pkl")
    path = root / str(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"bundle has no persisted BM25 index at {path.name}; "
            f"build Bm25Index(item_texts(catalogue)) instead")
    return Bm25Index.from_pickle(path)


def embedding_matrix(column: list) -> np.ndarray:
    """Turn a list-of-lists embedding column into a float64 matrix."""
    return np.asarray(column, dtype=np.float64)


def item_texts(catalogue: dict[str, list]) -> list[str]:
    """Title plus description, one string per catalogue row."""
    return [f"{title} {body}"
            for title, body in zip(catalogue["title"], catalogue["description"])]
