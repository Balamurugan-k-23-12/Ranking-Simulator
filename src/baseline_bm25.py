# Pure BM25 Baseline Ranker for Round 1
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add student helpers to path
STUDENT_DIR = Path(__file__).resolve().parent / "bundle_v1" / "student"
sys.path.insert(0, str(STUDENT_DIR))

from load import load_catalogue, load_bm25_index, load_schemas

def main():
    start_time = time.time()
    base_dir = Path(__file__).resolve().parent
    bundle_dir = base_dir / "bundle_v1"

    print("=== Step 1: Loading Schemas, Catalogue, and BM25 Index ===")
    schemas = load_schemas(bundle_dir)
    world_id = schemas["world_id"]
    print(f"World ID: {world_id}")

    cat = load_catalogue(bundle_dir, with_embeddings=False)
    item_ids = np.array(cat["item_id"])
    print(f"Loaded catalogue with {len(item_ids)} items.")

    t_bm25 = time.time()
    bm25 = load_bm25_index(bundle_dir)
    print(f"Loaded BM25 index in {time.time() - t_bm25:.2f}s (Vocabulary: {bm25.vocabulary_size} terms).")

    print("\n=== Step 2: Scoring 144 Unique Queries with BM25 ===")
    queries_df = pd.read_parquet(bundle_dir / "queries.parquet")
    print(f"Found {len(queries_df)} unique queries.")

    query_top20_map = {}
    for _, row in queries_df.iterrows():
        q_id = row["query_id"]
        q_text = row["text"]
        scores = bm25.scores(q_text)
        # Get top 20 items by highest BM25 score
        top_20_indices = np.argsort(-scores)[:20]
        query_top20_map[q_id] = item_ids[top_20_indices]

    print("Precomputed Top 20 BM25 items for all 144 queries.")

    print("\n=== Step 3: Generating Submission for Unique (Query, User) Pairs ===")
    traffic_file = base_dir / "traffic_r1.parquet"
    if not traffic_file.is_file():
        raise FileNotFoundError(f"traffic_r1.parquet not found at {traffic_file}")

    traffic_df = pd.read_parquet(traffic_file)
    unique_pairs_df = traffic_df[["query_id", "user_id"]].drop_duplicates()
    n_pairs = len(unique_pairs_df)
    print(f"Loaded traffic with {len(traffic_df)} impressions -> {n_pairs} unique (query_id, user_id) pairs.")

    # Build submission DataFrame
    rows_world_id = []
    rows_query_id = []
    rows_user_id = []
    rows_rank = []
    rows_item_id = []

    for _, pair in unique_pairs_df.iterrows():
        q_id = pair["query_id"]
        u_id = pair["user_id"]
        items = query_top20_map[q_id]
        
        rows_world_id.extend([world_id] * 20)
        rows_query_id.extend([q_id] * 20)
        rows_user_id.extend([u_id] * 20)
        rows_rank.extend(list(range(20)))
        rows_item_id.extend(items)

    submission_df = pd.DataFrame({
        "world_id": rows_world_id,
        "query_id": rows_query_id,
        "user_id": rows_user_id,
        "rank": np.array(rows_rank, dtype=np.int64),
        "item_id": rows_item_id
    })

    print(f"Generated submission with {len(submission_df)} rows.")

    print("\n=== Step 4: Strict Validation Checks ===")
    assert len(submission_df) == n_pairs * 20, f"Expected {n_pairs * 20} rows, got {len(submission_df)}"
    assert list(submission_df.columns) == ["world_id", "query_id", "user_id", "rank", "item_id"], "Invalid column names!"
    assert (submission_df["world_id"] == world_id).all(), "World ID mismatch!"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    assert (submission_df["rank"].min() == 0) and (submission_df["rank"].max() == 19), "Rank range invalid!"
    
    # Check no duplicate pairs have duplicate ranks
    pair_counts = submission_df.groupby(["query_id", "user_id"])["rank"].count()
    assert (pair_counts == 20).all(), "Some pairs do not have exactly 20 ranks!"

    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")

    out_path = base_dir / "submission_r1.parquet"
    submission_df.to_parquet(out_path, index=False)
    print(f"\nSaved submission to: {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Total time elapsed: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
