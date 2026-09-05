"""
BM25-only Ranker — Round 1
==========================
Uses only the BM25 retriever to rank each query and submits a pure lexical slate.
All session artifacts are saved locally ONLY under work/<session_id>/ — this is
the single local mount point for a session's inputs, submission, actions, and report.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

from retriever_bm25 import BM25Retriever

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
BUNDLE_DIR = ROOT_DIR / "bundle_v1"
WORK_DIR = ROOT_DIR / "work"

API_BASE = os.getenv("SIMULATOR_API_BASE", "http://127.0.0.1:8000")
WORLD_ID = "w908832361965e48e"
SLATE_SIZE = 20
ROUND_INDEX = 1

WORK_DIR.mkdir(parents=True, exist_ok=True)


def load_query_texts(bundle_dir: Path) -> dict[str, str]:
    q_table = pq.read_table(str(bundle_dir / "queries.parquet"), columns=["query_id", "text"])
    qids = q_table.column("query_id").to_pylist()
    texts = q_table.column("text").to_pylist()
    return dict(zip(qids, texts))


def build_bm25_slates(traffic_df: pd.DataFrame, bm25_retriever: BM25Retriever, query_texts: dict[str, str]) -> pd.DataFrame:
    unique_queries = traffic_df["query_id"].unique()
    print(f"\n[BM25-only] Ranking {len(unique_queries)} unique queries...")

    query_top20_items: dict[str, list[str]] = {}
    for i, qid in enumerate(unique_queries, start=1):
        q_text = query_texts.get(qid, "")
        idxs, _ = bm25_retriever.get_top_k(q_text, k=SLATE_SIZE)
        top20_item_ids = [bm25_retriever.item_ids[int(idx)] for idx in idxs[:SLATE_SIZE]]
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
    assert len(submission_df) == expected_rows, f"Expected {expected_rows}, got {len(submission_df)}"
    assert list(submission_df.columns) == ["world_id", "query_id", "user_id", "rank", "item_id"]
    assert not submission_df.isnull().any().any(), "Nulls found"
    print("✓ Validation passed.")


def delete_old_sessions() -> None:
    """Clear stale simulator sessions (container-side only) so a fresh run is not
    blocked by session limits. Never touches local work/ files."""
    try:
        list_resp = requests.get(f"{API_BASE}/v1/sessions", timeout=30)
        if list_resp.status_code != 200:
            print(f"[Session cleanup] Could not list sessions: status={list_resp.status_code} — skipping cleanup.")
            return

        payload = list_resp.json()
        session_ids: list[str] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    sid = item.get("session_id") or item.get("id")
                    if sid:
                        session_ids.append(str(sid))
        elif isinstance(payload, dict):
            for key in ("sessions", "items", "results"):
                if key in payload and isinstance(payload[key], list):
                    for item in payload[key]:
                        if isinstance(item, dict):
                            sid = item.get("session_id") or item.get("id")
                            if sid:
                                session_ids.append(str(sid))
                    break
            if not session_ids and "session_id" in payload:
                session_ids.append(str(payload["session_id"]))

        for sid in sorted(set(session_ids)):
            try:
                del_resp = requests.delete(f"{API_BASE}/v1/sessions/{sid}", timeout=30)
                if del_resp.status_code in (200, 202, 204):
                    print(f"[Session cleanup] Deleted stale session {sid}")
                else:
                    print(f"[Session cleanup] Could not delete {sid}: {del_resp.status_code} {del_resp.text[:120]}")
            except Exception as exc:
                print(f"[Session cleanup] Error deleting {sid}: {exc}")
    except Exception as exc:
        print(f"[Session cleanup] Error listing sessions: {exc}")


def fetch_report_with_retry(session_id: str, max_attempts: int = 5, wait_seconds: int = 15):
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
    print("      ROUND 1: BM25-ONLY EXPERIMENT                       ")
    print("==========================================================")
    start_time = time.time()

    print("[1/7] Checking simulator health...")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=30)
        if health.status_code != 200:
            print(f"Health failed: {health.status_code}")
            return
        print(f"Health OK: {health.json().get('world_id')}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    print("[2/7] Cleaning stale simulator sessions (container-side only)...")
    delete_old_sessions()

    print("[3/7] Loading BM25 retriever...")
    bm25_retriever = BM25Retriever(BUNDLE_DIR)
    query_texts = load_query_texts(BUNDLE_DIR)

    print("[4/7] Opening simulator session...")
    sess_resp = requests.post(f"{API_BASE}/v1/sessions?seed=42", timeout=30)
    if sess_resp.status_code != 200:
        print(f"Session failed: {sess_resp.text}")
        return
    session_id = sess_resp.json()["session_id"]
    session_dir = WORK_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session started: {session_id}")
    print(f"Mounted local work folder: {session_dir}")

    # All per-session files live directly under session_dir from here on
    traffic_file = session_dir / "traffic_r1.parquet"
    submission_file = session_dir / "submission_bm25_only_r1.parquet"
    actions_file = session_dir / "actions_bm25_only_r1.parquet"
    report_file = session_dir / "report_bm25_only_r1.json"

    session_meta = {
        "session_id": session_id,
        "world_id": WORLD_ID,
        "round": ROUND_INDEX,
        "created_at": time.time(),
        "status": "initialized",
        "model": "bm25_only_retriever",
    }
    (session_dir / "session.json").write_text(json.dumps(session_meta, indent=2))

    print("[4/7] Fetching traffic...")
    traf_resp = requests.get(f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/traffic", timeout=60)
    if traf_resp.status_code == 200:
        traffic_file.write_bytes(traf_resp.content)
        print(f"Downloaded traffic ({len(traf_resp.content)/1024:.1f} KB) -> {traffic_file}")
    else:
        print(f"Traffic download failed: {traf_resp.status_code} {traf_resp.text}")
        return

    traffic_raw = pd.read_parquet(traffic_file)
    traffic = traffic_raw.drop_duplicates(subset=["query_id", "user_id"]).reset_index(drop=True)

    print("[5/7] Building BM25 slates...")
    submission = build_bm25_slates(traffic, bm25_retriever, query_texts)
    validate_submission(submission, len(traffic))
    submission.to_parquet(submission_file, index=False)
    print(f"Saved {submission_file}")

    print("[6/7] Submitting BM25-only slates...")
    with open(submission_file, "rb") as f:
        up_resp = requests.post(
            f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/submission",
            files={"file": (submission_file.name, f, "application/octet-stream")},
            timeout=None,
        )
    if up_resp.status_code != 200:
        print(f"Upload failed: {up_resp.status_code} {up_resp.text}")
        return
    print(f"Submission accepted: {up_resp.json()}")
    (session_dir / "submission_response.json").write_text(json.dumps(up_resp.json(), indent=2))

    print("[7/7] Downloading actions and report...")
    act_resp = requests.get(f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/actions", timeout=300)
    if act_resp.status_code == 200:
        actions_file.write_bytes(act_resp.content)
        print(f"Saved actions log: {actions_file} ({len(act_resp.content)/1024:.1f} KB)")
    else:
        print(f"Actions download failed: {act_resp.status_code} {act_resp.text}")

    rep_resp = fetch_report_with_retry(session_id)
    report_data = None
    if rep_resp is not None and rep_resp.status_code == 200:
        report_file.write_bytes(rep_resp.content)
        report_data = rep_resp.json()
        score_data = {
            "session_id": session_id,
            "world_id": report_data.get("world_id", WORLD_ID),
            "round": report_data.get("round", ROUND_INDEX),
            "n_pairs": report_data.get("n_pairs"),
            "n_impressions": report_data.get("n_impressions"),
            "expected_value": report_data.get("expected_value"),
            "sampled_value": report_data.get("sampled_value"),
            "action_counts": report_data.get("action_counts"),
        }
        (session_dir / "score.json").write_text(json.dumps(score_data, indent=2))
        print(json.dumps(report_data, indent=2))
    else:
        print(f"Could not fetch report — session {session_id} left alive on simulator "
              f"so you can retry fetching it later.")

    config = {
        "model": "BM25OnlyRetriever",
        "method": "BM25 Lexical Ranking",
        "round": ROUND_INDEX,
        "session_id": session_id,
    }
    (session_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print(f"\nAll artifacts successfully saved under work/{session_id}/")
    print(f"Total time elapsed: {time.time() - start_time:.2f}s")

    if report_data is not None:
        try:
            requests.delete(f"{API_BASE}/v1/sessions/{session_id}", timeout=30)
            print(f"Successfully deleted session {session_id} from simulator.")
        except Exception as e:
            print(f"Warning: Failed to delete session {session_id}: {e}")


if __name__ == "__main__":
    main()