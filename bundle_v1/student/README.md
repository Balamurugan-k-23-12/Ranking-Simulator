# Student data bundle helpers

This ``student/`` package ships inside ``bundle_v1/``. It has no dependency on the
course repositories (usim, coms, mplc). You need Python 3.11+, numpy, and pyarrow.

## Layout

| Path | Contents |
|---|---|
| ``catalogue.parquet`` | public items + embedding, stamped with ``world_id`` |
| ``users.parquet`` | ``user_id``, ``segment``, observed features |
| ``users_heldout.parquet`` | same schema, fresh users from the same process (prefix ``H``) |
| ``queries.parquet`` | public queries (no intent link) |
| ``bootstrap_log.parquet`` | starter action log: correct schema, zero rows |
| ``query_embeddings.parquet`` | precomputed ``e_q`` for the published queries |
| ``schemas.json`` | column lists, embedder identity, BM25 params |
| ``student/`` | this package: load, embedder, BM25, example |

## Quick start

From the ``bundle_v1/`` directory:

```
PYTHONPATH=student python student/example.py
```

Or in your own code:

```python
from load import load_catalogue, load_schemas, item_texts
from embedder import EmbedderConfig, MockHashEmbedder
from bm25 import Bm25Index

bundle = "."   # the bundle_v1 directory
schemas = load_schemas(bundle)
catalogue = load_catalogue(bundle)
embedder = MockHashEmbedder(EmbedderConfig.from_schemas(schemas))
index = Bm25Index(item_texts(catalogue),
                  k1=schemas["bm25"]["k1"], b=schemas["bm25"]["b"])

e_q = embedder.encode(["wireless earbuds"])[0]
scores = index.scores("wireless earbuds")
```

The mock embedder rebuilds the same space as the catalogue embeddings, so cosine
search over ``embedding`` matches the semantic channel of the course. The
``query_embeddings.parquet`` file holds the same vectors for the published query texts.
