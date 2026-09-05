"""
Semantic Baseline — Round 2
============================
Ranks 100k items for every (query, user) pair using COSINE SIMILARITY
between pre-computed query embeddings and item embeddings.

No AI model download needed — the vectors are already in the bundle!

How it works:
  1. Load item embeddings from catalogue.parquet   → shape (100000, 128)
  2. Load query embeddings from query_embeddings.parquet → shape (144, 128)
  3. For each query: cosine_similarity = query_vec · item_vec  (dot product of unit vectors)
  4. Rank items top-20 per (query_id, user_id) pair
  5. Export submission.parquet and POST to simulator
"""

import sys
import requests
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/Users/balamsd/Desktop/Assignment-1")
BUNDLE_DIR  = BASE_DIR / "bundle_v1"
STUDENT_DIR = BUNDLE_DIR / "student"
sys.path.insert(0, str(STUDENT_DIR))

TRAFFIC_FILE    = BASE_DIR / "traffic_r2.parquet"   # update round number as needed
SUBMISSION_FILE = BASE_DIR / "submission_semantic.parquet"
API_BASE        = "http://127.0.0.1:8000"
WORLD_ID        = "w908832361965e48e"
SLATE_SIZE      = 20

# ── step 1: load traffic ─────────────────────────────────────────────────────
print("Loading traffic file …")
traffic = pd.read_parquet(TRAFFIC_FILE)
print(f"  Raw rows: {len(traffic)}")

# Deduplicate (same fix as BM25 baseline — simulator rejects duplicate pairs)
traffic = traffic.drop_duplicates(subset=["query_id", "user_id"]).reset_index(drop=True)
print(f"  Unique (query_id, user_id) pairs: {len(traffic)}")

# ── step 2: load item embeddings ──────────────────────────────────────────────
print("Loading catalogue embeddings …")
cat_table = pq.read_table(str(BUNDLE_DIR / "catalogue.parquet"))
item_ids   = cat_table.column("item_id").to_pylist()
# embedding column is a list-of-lists; convert to numpy matrix (100000 × 128)
item_vecs  = np.asarray(cat_table.column("embedding").to_pylist(), dtype=np.float64)

# Unit-normalise item vectors so dot-product == cosine similarity
item_norms = np.linalg.norm(item_vecs, axis=1, keepdims=True)
item_vecs  = item_vecs / np.where(item_norms > 0, item_norms, 1.0)
print(f"  Item matrix: {item_vecs.shape}")   # expect (100000, 128)

# ── step 3: load query embeddings ─────────────────────────────────────────────
print("Loading query embeddings …")
qe_table   = pq.read_table(str(BUNDLE_DIR / "query_embeddings.parquet"))
qe_qids    = qe_table.column("query_id").to_pylist()
qe_vecs    = np.asarray(qe_table.column("embedding").to_pylist(), dtype=np.float64)

# Unit-normalise
qe_norms   = np.linalg.norm(qe_vecs, axis=1, keepdims=True)
qe_vecs    = qe_vecs / np.where(qe_norms > 0, qe_norms, 1.0)

# Build a lookup: query_id → query vector
qid_to_vec = {qid: qe_vecs[i] for i, qid in enumerate(qe_qids)}
print(f"  Query embeddings loaded: {len(qid_to_vec)} queries")

# ── step 4: score & rank ──────────────────────────────────────────────────────
print("Scoring and ranking …")
rows = []

# Cache: query_id → top-20 item indices (same for all users of the same query)
top20_cache: dict[str, list[int]] = {}

for _, tr in traffic.iterrows():
    qid = tr["query_id"]
    uid = tr["user_id"]

    if qid not in top20_cache:
        q_vec  = qid_to_vec.get(qid)
        if q_vec is None:
            # Fallback: score 0 for unknown query → random items
            top20 = list(range(SLATE_SIZE))
        else:
            # Cosine similarity = dot product (both vectors already unit-normed)
            scores  = item_vecs @ q_vec          # shape (100000,)
            top20   = np.argpartition(scores, -SLATE_SIZE)[-SLATE_SIZE:]  # fast top-k
            top20   = top20[np.argsort(scores[top20])[::-1]]              # sort desc
            top20   = top20.tolist()
        top20_cache[qid] = top20

    for rank, idx in enumerate(top20_cache[qid]):
        rows.append({
            "world_id": WORLD_ID,
            "query_id": qid,
            "user_id":  uid,
            "rank":     rank,
            "item_id":  item_ids[idx],
        })

submission = pd.DataFrame(rows)
print(f"  Submission rows: {len(submission)}")
print(f"  Expected:        {len(traffic) * SLATE_SIZE}")
assert len(submission) == len(traffic) * SLATE_SIZE, "Row count mismatch!"

# ── step 5: save submission parquet ──────────────────────────────────────────
submission.to_parquet(SUBMISSION_FILE, index=False)
print(f"\nSaved → {SUBMISSION_FILE}")

# ── step 6: submit to simulator ───────────────────────────────────────────────
print("\nStarting simulator session …")
sess_resp = requests.post(f"{API_BASE}/v1/sessions?seed=42")
if sess_resp.status_code != 200:
    print(f"Session creation failed: {sess_resp.status_code} {sess_resp.text}")
    sys.exit(1)

session_id = sess_resp.json()["session_id"]
print(f"  session_id = {session_id}")

print("Uploading submission …")
with open(SUBMISSION_FILE, "rb") as f:
    up_resp = requests.post(
        f"{API_BASE}/v1/sessions/{session_id}/submission",
        files={"file": ("submission_semantic.parquet", f, "application/octet-stream")},
    )

if up_resp.status_code != 200:
    print(f"Upload failed: {up_resp.status_code}")
    print(up_resp.text)
    sys.exit(1)

print("Upload accepted!")
print(up_resp.json())
