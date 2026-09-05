"""
Semantic-only Ranker — Round 1
=============================
Uses only the semantic retriever (cosine similarity) to score and rank each query.
Stores ALL session artifacts locally under work/<session_id>/ — this folder is the
single local mount point for a session's inputs, submission, actions, and report.
"""

from __future__ import annotations
import json
import os
import shutil
import time
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import requests

from retriever_semantic import SemanticRetriever

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
BUNDLE_DIR = ROOT_DIR / "bundle_v1"
WORK_DIR = ROOT_DIR / "work"

API_BASE = os.getenv("SIMULATOR_API_BASE", "http://127.0.0.1:8000")
WORLD_ID = "w908832361965e48e"
SLATE_SIZE = 20
ROUND_INDEX = 1

WORK_DIR.mkdir(parents=True, exist_ok=True)


def clear_container_sessions() -> None:
    """Clear stale sessions on the simulator side (container), not local files.

    Tries a bulk-clear endpoint first; falls back to listing + deleting
    individual sessions if the bulk endpoint isn't available.
    """
    try:
        resp = requests.delete(f"{API_BASE}/v1/sessions", timeout=30)
        if resp.status_code == 200:
            print(f"  Cleared all sessions on simulator: {resp.json()}")
            return
    except Exception as e:
        print(f"  Bulk session clear not available ({e}), trying per-session cleanup...")

    try:
        list_resp = requests.get(f"{API_BASE}/v1/sessions", timeout=30)
        if list_resp.status_code == 200:
            sessions = list_resp.json().get("sessions", [])
            for s in sessions:
                sid = s.get("session_id") if isinstance(s, dict) else s
                try:
                    requests.delete(f"{API_BASE}/v1/sessions/{sid}", timeout=30)
                    print(f"  Deleted stale container session: {sid}")
                except Exception as e:
                    print(f"  Warning: failed to delete session {sid}: {e}")
        else:
            print(f"  Could not list sessions ({list_resp.status_code}); skipping cleanup.")
    except Exception as e:
        print(f"  Warning: session cleanup skipped: {e}")


def build_semantic_slates(traffic_df: pd.DataFrame, sem_retriever: SemanticRetriever) -> pd.DataFrame:
    unique_queries = traffic_df["query_id"].unique()
    print(f"\n[Semantic-only] Ranking {len(unique_queries)} unique queries...")

    query_top20_items: dict[str, list[str]] = {}
    for i, qid in enumerate(unique_queries, start=1):
        idxs, _ = sem_retriever.get_top_k(qid, k=SLATE_SIZE)
        top20_item_ids = [sem_retriever.item_ids[int(idx)] for idx in idxs[:SLATE_SIZE]]
        query_top20_items[qid] = top20_item_ids

        if i % 30 == 0 or i == len(unique_queries):
            print(f"  Processed {i}/{len(unique_queries)} queries...")

    rows = []
    for _, tr in traffic_df.iterrows():
        qid = tr["query_id"]
        uid = tr["user_id"]
        for rank, item_id in enumerate(query_top20_items[qid]):
            rows.append({
                "world_id": WORLD_ID,
                "query_id": qid,
                "user_id": uid,
                "rank": rank,
                "item_id": item_id,
            })

    return pd.DataFrame(rows)


def validate_submission(submission_df: pd.DataFrame, expected_pairs_count: int) -> None:
    expected_rows = expected_pairs_count * SLATE_SIZE
    assert len(submission_df) == expected_rows, f"Expected {expected_rows} rows, got {len(submission_df)}"
    assert list(submission_df.columns) == ["world_id", "query_id", "user_id", "rank", "item_id"]
    assert not submission_df.isnull().any().any(), "Submission contains null values!"
    print(f"  ✓ Validation passed: {len(submission_df):,} rows verified.")


def archive_inputs(session_dir: Path) -> None:
    inputs_dir = session_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for name in ["catalogue.parquet", "queries.parquet", "query_embeddings.parquet", "users.parquet"]:
        source = BUNDLE_DIR / name
        if source.exists():
            shutil.copy2(source, inputs_dir / name)


def main():
    print("==========================================================")
    print("      ROUND 1: PURE SEMANTIC RETRIEVAL EXPERIMENT         ")
    print("==========================================================")
    start_time = time.time()

    # 0. Clear stale sessions on the simulator container (local work/ files are untouched)
    print(f"\n[0/6] Clearing old sessions on simulator container...")
    clear_container_sessions()

    # 1. Health check
    print(f"\n[1/6] Checking simulator health at {API_BASE}...")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=30)
        if health.status_code != 200:
            print(f"  Health failed: {health.status_code}")
            return
        print(f"  Simulator Healthy! World ID: {health.json().get('world_id')}")
    except Exception as e:
        print(f"  Health check failed: {e}")
        return

    # 2. Pre-load Semantic Retriever
    print("\n[2/6] Loading semantic retriever...")
    sem_retriever = SemanticRetriever(BUNDLE_DIR)

    # 3. Open session & create work/<session_id> folder — this IS the local mount point
    print(f"\n[3/6] Opening simulator session...")
    sess_resp = requests.post(f"{API_BASE}/v1/sessions?seed=42", timeout=30)
    if sess_resp.status_code != 200:
        print(f"  Session creation failed: {sess_resp.text}")
        return
    session_id = sess_resp.json()["session_id"]
    session_dir = WORK_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # All per-session files live directly under session_dir from here on
    traffic_file = session_dir / "traffic_semantic_r1.parquet"
    submission_file = session_dir / "submission_semantic_only_r1.parquet"
    actions_file = session_dir / "actions_semantic_only_r1.parquet"
    report_file = session_dir / "report_semantic_only_r1.json"

    session_meta = {
        "session_id": session_id,
        "world_id": WORLD_ID,
        "round": ROUND_INDEX,
        "created_at": time.time(),
        "status": "initialized",
        "model": "pure_semantic_retriever",
    }
    (session_dir / "session.json").write_text(json.dumps(session_meta, indent=2))
    archive_inputs(session_dir)
    print(f"  Session started: session_id = {session_id}")
    print(f"  Mounted local work folder: {session_dir}")

    # 4. Fetch traffic — save straight into session_dir
    print(f"\n[4/6] Fetching Round {ROUND_INDEX} traffic...")
    traf_resp = requests.get(f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/traffic", timeout=60)
    if traf_resp.status_code == 200:
        traffic_file.write_bytes(traf_resp.content)
        print(f"  Downloaded traffic ({len(traf_resp.content)/1024:.1f} KB) -> {traffic_file}")
    else:
        print(f"  Traffic download failed: {traf_resp.status_code} {traf_resp.text}")
        return

    traffic_raw = pd.read_parquet(traffic_file)
    traffic = traffic_raw.drop_duplicates(subset=["query_id", "user_id"]).reset_index(drop=True)
    print(f"  Unique (query_id, user_id) pairs to rank: {len(traffic):,}")

    # 5. Build semantic slates
    print("\n[5/6] Building semantic slates...")
    submission = build_semantic_slates(traffic, sem_retriever)
    validate_submission(submission, len(traffic))
    submission.to_parquet(submission_file, index=False)
    print(f"  Saved submission to {submission_file}")

    # 6. Submit to simulator
    print("\n[6/6] Submitting semantic-only slates to simulator...")
    with open(submission_file, "rb") as f:
        up_resp = requests.post(
            f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/submission",
            files={"file": (submission_file.name, f, "application/octet-stream")},
            timeout=None,
        )
    if up_resp.status_code != 200:
        print(f"  Upload failed ({up_resp.status_code}): {up_resp.text}")
        return
    print(f"  Submission accepted! Response: {up_resp.json()}")
    submission_response = up_resp.json()
    (session_dir / "submission_response.json").write_text(json.dumps(submission_response, indent=2))

    # Download Actions
    print("  Downloading simulated user action logs...")
    act_resp = requests.get(f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/actions", timeout=300)
    if act_resp.status_code == 200:
        actions_file.write_bytes(act_resp.content)
        print(f"  Saved actions log: {actions_file} ({len(act_resp.content)/1024:.1f} KB)")

    # Download Report
    print("  Downloading official evaluation report...")
    rep_resp = requests.get(f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/report", timeout=300)
    if rep_resp.status_code == 200:
        report_file.write_bytes(rep_resp.content)
        report_data = rep_resp.json()

        score_data = {
            "session_id": session_id,
            "world_id": report_data.get("world_id", WORLD_ID),
            "round": report_data.get("round", ROUND_INDEX),
            "expected_value": report_data.get("expected_value"),
            "sampled_value": report_data.get("sampled_value"),
            "action_counts": report_data.get("action_counts"),
            "n_impressions": report_data.get("n_impressions"),
        }
        (session_dir / "score.json").write_text(json.dumps(score_data, indent=2))

        print("\n==========================================================")
        print("       OFFICIAL ROUND 1 SEMANTIC EVALUATION REPORT        ")
        print("==========================================================")
        print(json.dumps(report_data, indent=2))
        print("==========================================================")

    # Save run config
    config = {
        "model": "PureSemanticRetriever",
        "method": "Dense Vector Cosine Similarity",
        "embedding_dim": 384,
        "round": ROUND_INDEX,
        "session_id": session_id,
    }
    (session_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print(f"\nAll artifacts successfully saved under work/{session_id}/")
    print(f"Total time elapsed: {time.time() - start_time:.2f}s")

    try:
        requests.delete(f"{API_BASE}/v1/sessions/{session_id}", timeout=30)
        print(f"Successfully deleted session {session_id} from simulator.")
    except Exception as e:
        print(f"Warning: Failed to delete session {session_id}: {e}")


if __name__ == "__main__":
    main()