"""
Hybrid Reciprocal Rank Fusion (RRF) Ranker — Round 1
=====================================================
Combines:
  1. BM25 Keyword Search (retriever_bm25.py)
  2. Dense Semantic Vector Cosine Similarity (retriever_semantic.py)
  3. Product Quality / Rating Metadata (scorer_quality.py)

Uses 3-Way Reciprocal Rank Fusion (RRF) with k=60 to fuse ranks into a superior slate.

All session artifacts are saved locally ONLY under work/<session_id>/ — this is the
single local mount point for a session's inputs, submission, actions, and report.
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

# Import modular components
from retriever_bm25 import BM25Retriever
from retriever_semantic import SemanticRetriever
from scorer_quality import QualityScorer

# ── Paths & Constants ────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).resolve().parent        # src/
ROOT_DIR = SRC_DIR.parent                          # assignment-1/
BUNDLE_DIR = ROOT_DIR / "bundle_v1"
WORK_DIR   = ROOT_DIR / "work"

WORK_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = os.getenv("SIMULATOR_API_BASE", "http://127.0.0.1:8000")
WORLD_ID = "w908832361965e48e"
SLATE_SIZE = 20
RRF_K = 60.0
TOP_CANDIDATES = 500  # Number of candidates retrieved per channel
QUALITY_WEIGHT = 0.2  # Weight for product quality rank in fusion
ROUND_INDEX = 1


def load_query_texts(bundle_dir: Path) -> dict[str, str]:
    """Loads query_id -> text mapping."""
    q_table = pq.read_table(str(bundle_dir / "queries.parquet"), columns=["query_id", "text"])
    qids = q_table.column("query_id").to_pylist()
    texts = q_table.column("text").to_pylist()
    return dict(zip(qids, texts))


def build_hybrid_slates(
    traffic_df: pd.DataFrame,
    bm25_retriever: BM25Retriever,
    sem_retriever: SemanticRetriever,
    quality_scorer: QualityScorer,
    query_texts: dict[str, str],
) -> pd.DataFrame:
    """
    Computes top-20 item slates for all unique (query_id, user_id) pairs using 3-Way RRF.
    """
    unique_queries = traffic_df["query_id"].unique()
    print(f"\n[Hybrid Pipeline] Computing 3-Way RRF for {len(unique_queries)} unique queries...")

    # Cache query_id -> list of top 20 item_ids
    query_top20_items: dict[str, list[str]] = {}

    for i, qid in enumerate(unique_queries, start=1):
        q_text = query_texts.get(qid, "")

        # 1. Retrieve BM25 candidates
        bm25_indices, _ = bm25_retriever.get_top_k(q_text, k=TOP_CANDIDATES)
        bm25_ranks = {int(idx): rank for rank, idx in enumerate(bm25_indices, start=1)}

        # 2. Retrieve Semantic candidates
        sem_indices, _ = sem_retriever.get_top_k(qid, k=TOP_CANDIDATES)
        sem_ranks = {int(idx): rank for rank, idx in enumerate(sem_indices, start=1)}

        # 3. Union of candidates
        candidate_indices = list(set(bm25_ranks.keys()) | set(sem_ranks.keys()))

        # 4. Get quality ranks within candidate pool
        quality_ranks = quality_scorer.get_quality_ranks_for_candidates(candidate_indices)

        # 5. Compute 3-Way RRF Score for each candidate
        scored_candidates = []
        for idx in candidate_indices:
            r_bm25 = bm25_ranks.get(idx)
            r_sem = sem_ranks.get(idx)
            r_qual = quality_ranks.get(idx, len(candidate_indices))

            score = 0.0
            if r_bm25 is not None:
                score += 1.0 / (RRF_K + r_bm25)
            if r_sem is not None:
                score += 1.0 / (RRF_K + r_sem)
            if r_qual is not None:
                score += QUALITY_WEIGHT / (RRF_K + r_qual)

            scored_candidates.append((score, idx))

        # Sort by RRF score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top20_idx = [idx for _, idx in scored_candidates[:SLATE_SIZE]]
        top20_item_ids = [bm25_retriever.item_ids[idx] for idx in top20_idx]

        query_top20_items[qid] = top20_item_ids

        if i % 30 == 0 or i == len(unique_queries):
            print(f"  Processed {i}/{len(unique_queries)} queries...")

    # Assemble submission rows for each traffic impression
    print("\n[Hybrid Pipeline] Assembling submission DataFrame...")
    rows = []
    for _, tr in traffic_df.iterrows():
        qid = tr["query_id"]
        uid = tr["user_id"]
        slate = query_top20_items[qid]

        for rank, item_id in enumerate(slate):
            rows.append({
                "world_id": WORLD_ID,
                "query_id": qid,
                "user_id": uid,
                "rank": rank,
                "item_id": item_id,
            })

    submission_df = pd.DataFrame(rows)
    return submission_df


def validate_submission(submission_df: pd.DataFrame, expected_pairs_count: int) -> None:
    """Validates strict submission constraints."""
    print("\n[Validation] Checking submission format...")
    expected_rows = expected_pairs_count * SLATE_SIZE
    assert len(submission_df) == expected_rows, f"Expected {expected_rows} rows, got {len(submission_df)}"

    expected_cols = ["world_id", "query_id", "user_id", "rank", "item_id"]
    assert list(submission_df.columns) == expected_cols, f"Columns mismatch: {submission_df.columns}"

    assert not submission_df.isnull().any().any(), "Submission contains null values!"
    print(f"  ✓ Row count verified: {len(submission_df):,} rows.")
    print(f"  ✓ Schema and non-null verified: {expected_cols}")
    print("  ✓ All validation checks passed!")


def fetch_report_with_retry(session_id: str, max_attempts: int = 8, wait_seconds: int = 20):
    """Retry the report endpoint since the simulator may still be aggregating results."""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(
                f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/report",
                timeout=300,
            )
            return resp
        except requests.exceptions.ReadTimeout:
            print(f"  Report fetch timed out (attempt {attempt}/{max_attempts}); "
                  f"simulator may still be aggregating. Retrying in {wait_seconds}s...")
        except requests.exceptions.RequestException as e:
            print(f"  Report fetch failed (attempt {attempt}/{max_attempts}): {e}. Retrying in {wait_seconds}s...")
        time.sleep(wait_seconds)
    return None


def main():
    print("==========================================================")
    print("      ROUND 1: 3-WAY RECIPROCAL RANK FUSION (RRF)         ")
    print("==========================================================")
    start_time = time.time()

    # ── PHASE 1: Pre-load ALL heavy modules BEFORE opening a session ──────────
    # The simulator kills the session if submission takes too long after traffic
    # fetch. Load everything first so the session window only does fast work.
    print(f"\n[1/6] Checking simulator health at {API_BASE}...")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=30)
        if health.status_code != 200:
            print(f"  Simulator health check failed ({health.status_code}).")
            return
        print(f"  Simulator Healthy! World ID: {health.json().get('world_id')}")
    except requests.exceptions.RequestException as e:
        print(f"  Simulator API error: {e}")
        return

    print("\n[2/6] Pre-loading BM25, Semantic & Quality modules (before session open)...")
    bm25_retriever = BM25Retriever(BUNDLE_DIR)
    sem_retriever = SemanticRetriever(BUNDLE_DIR)
    quality_scorer = QualityScorer(BUNDLE_DIR)
    query_texts = load_query_texts(BUNDLE_DIR)
    print(f"  All modules loaded in {time.time() - start_time:.1f}s — ready for fast session!")

    # ── PHASE 2: Open session → fetch traffic → build slates → submit FAST ───
    print(f"\n[3/6] Opening Simulator Session...")
    sess_resp = requests.post(f"{API_BASE}/v1/sessions?seed=42", timeout=30)
    if sess_resp.status_code != 200:
        print(f"  Failed to create session: {sess_resp.text}")
        return
    session_id = sess_resp.json()["session_id"]
    session_dir = WORK_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Session started: session_id = {session_id}")
    print(f"  Mounted local work folder: {session_dir}")

    # All per-session files live directly under session_dir from here on
    traffic_file = session_dir / "traffic_r1.parquet"
    submission_file = session_dir / "submission_hybrid_r1.parquet"
    actions_file = session_dir / "actions_r1.parquet"
    report_file = session_dir / "report_hybrid_r1.json"

    (session_dir / "session.json").write_text(json.dumps({
        "session_id": session_id,
        "world_id": WORLD_ID,
        "round": ROUND_INDEX,
        "created_at": time.time(),
        "status": "initialized",
        "model": "hybrid_rrf_bm25_semantic_quality",
    }, indent=2))

    t_session = time.time()
    print(f"\n[4/6] Fetching Round {ROUND_INDEX} traffic from simulator...")
    traf_resp = requests.get(
        f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/traffic",
        timeout=60,
    )
    if traf_resp.status_code == 200:
        traffic_file.write_bytes(traf_resp.content)
        print(f"  Downloaded {traffic_file.name} ({len(traf_resp.content)/1024:.1f} KB) -> {traffic_file}")
    else:
        print(f"  Could not download traffic ({traf_resp.status_code}): {traf_resp.text}")
        return

    traffic_raw = pd.read_parquet(traffic_file)
    traffic = traffic_raw.drop_duplicates(subset=["query_id", "user_id"]).reset_index(drop=True)
    print(f"  Unique (query_id, user_id) impressions to rank: {len(traffic):,}")

    print("\n[5/6] Building Hybrid Slates & Validating...")
    submission = build_hybrid_slates(
        traffic_df=traffic,
        bm25_retriever=bm25_retriever,
        sem_retriever=sem_retriever,
        quality_scorer=quality_scorer,
        query_texts=query_texts,
    )
    validate_submission(submission, len(traffic))
    submission.to_parquet(submission_file, index=False)
    print(f"  Saved submission parquet: {submission_file.stat().st_size / 1024:.1f} KB -> {submission_file}")
    print(f"  Time since session open: {time.time()-t_session:.1f}s")

    # 5. Submit to Simulator
    print(f"\n[6/6] Submitting Round {ROUND_INDEX} Slates to Simulator API...")
    with open(submission_file, "rb") as f:
        up_resp = requests.post(
            f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/submission",
            files={"file": (submission_file.name, f, "application/octet-stream")},
            timeout=None,  # Simulator scoring can take 5-10 min; never drop connection
        )
    if up_resp.status_code != 200:
        print(f"  Submission failed ({up_resp.status_code}): {up_resp.text}")
        return
    print(f"  Submission accepted! Response: {up_resp.json()}")
    (session_dir / "submission_response.json").write_text(json.dumps(up_resp.json(), indent=2))

    # Download Actions & Report
    print("\n  Downloading simulation actions and report...")
    act_resp = requests.get(f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/actions", timeout=300)
    if act_resp.status_code == 200:
        actions_file.write_bytes(act_resp.content)
        print(f"  Saved actions log: {actions_file} ({len(act_resp.content)/1024:.1f} KB)")
    else:
        print(f"  Actions download failed: {act_resp.status_code} {act_resp.text}")

    rep_resp = fetch_report_with_retry(session_id)
    report_data = None
    if rep_resp is None:
        print(f"  [Report] All retry attempts exhausted — no response received at all "
              f"(likely network/timeout). Session {session_id} left alive.")
    elif rep_resp.status_code != 200:
        print(f"  [Report] Got HTTP {rep_resp.status_code} instead of 200: {rep_resp.text[:300]}")
        print(f"  Session {session_id} left alive — report may not be ready yet, or round/session id mismatch.")
    else:
        report_file.write_bytes(rep_resp.content)
        report_data = rep_resp.json()
        score_payload = {
            "session_id": session_id,
            "world_id": report_data.get("world_id", WORLD_ID),
            "round": report_data.get("round", ROUND_INDEX),
            "n_pairs": report_data.get("n_pairs"),
            "n_impressions": report_data.get("n_impressions"),
            "expected_value": report_data.get("expected_value"),
            "sampled_value": report_data.get("sampled_value"),
            "action_counts": report_data.get("action_counts"),
        }
        (session_dir / "score.json").write_text(json.dumps(score_payload, indent=2))
        print("\n==========================================================")
        print("            OFFICIAL ROUND 1 EVALUATION REPORT            ")
        print("==========================================================")
        print(json.dumps(report_data, indent=2))
        print("==========================================================")

    config = {
        "model": "HybridRRF",
        "method": "3-Way Reciprocal Rank Fusion (BM25 + Semantic + Quality)",
        "rrf_k": RRF_K,
        "top_candidates": TOP_CANDIDATES,
        "quality_weight": QUALITY_WEIGHT,
        "round": ROUND_INDEX,
        "session_id": session_id,
    }
    (session_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print(f"\nAll artifacts successfully saved under work/{session_id}/")
    print(f"Total time elapsed: {time.time() - start_time:.2f}s")

    if report_data is not None:
        print(f"[Cleanup] Report received successfully — deleting session {session_id}...")
        try:
            del_resp = requests.delete(f"{API_BASE}/v1/sessions/{session_id}", timeout=30)
            if del_resp.status_code in (200, 202, 204):
                print(f"[Cleanup] Successfully deleted session {session_id} from simulator.")
            else:
                print(f"[Cleanup] Delete call returned {del_resp.status_code}: {del_resp.text[:200]}")
        except Exception as e:
            print(f"[Cleanup] Warning: Failed to delete session {session_id}: {e}")
    else:
        print(f"[Cleanup] Skipped — report was not retrieved, so session {session_id} "
              f"remains alive on the simulator. Retry fetching it later, then delete manually if needed.")


if __name__ == "__main__":
    main()