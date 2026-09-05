"""Round 2 LambdaMART submission using Round 1 simulator feedback.

IMPORTANT: This round reuses the SAME session_id that Round 1 (the hybrid RRF
script) opened. It does NOT create a new session. This keeps Round 1 and
Round 2 tied to one session on the simulator and one folder locally:
work/<session_id>/.

You must pass the Round 1 session_id explicitly, either as:
  - a CLI argument:  python ltr_round2.py <session_id>
  - or an env var:   SESSION_ID=<session_id> python ltr_round2.py

The session_id must correspond to a session that is still alive on the
simulator (i.e. Round 1's script did not delete it) and for which
work/<session_id>/ already exists locally with Round 1's artifacts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import requests

from ltr_feature_builder import FEATURE_COLUMNS, LTRFeatureBuilder, action_to_weight
from ltr_model import LambdaMartReranker, build_group_sizes
from retriever_bm25 import BM25Retriever
from retriever_semantic import SemanticRetriever

ROOT_DIR = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT_DIR / "bundle_v1"
WORK_DIR = ROOT_DIR / "work"
API_BASE = os.getenv("SIMULATOR_API_BASE", "http://127.0.0.1:8000")
WORLD_ID = "w908832361965e48e"
ROUND_INDEX = 2
TOP_CANDIDATES = 5000
SLATE_SIZE = 20

WORK_DIR.mkdir(parents=True, exist_ok=True)


def resolve_session_id() -> str:
    """Round 2 must reuse Round 1's session_id — never create a new one."""
    session_id = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SESSION_ID")
    if not session_id:
        print("ERROR: You must supply the Round 1 session_id to reuse for Round 2.")
        print("  Usage: python ltr_round2.py <session_id>")
        print("  Or:    SESSION_ID=<session_id> python ltr_round2.py")
        sys.exit(1)

    session_dir = WORK_DIR / session_id
    if not session_dir.exists():
        print(f"ERROR: work/{session_id}/ not found locally. This should be the same "
              f"folder Round 1 (hybrid) created. Did you pass the right session_id?")
        sys.exit(1)

    round1_submission = session_dir / "submission_hybrid_r1.parquet"
    round1_actions = session_dir / "actions_r1.parquet"
    if not round1_submission.exists() or not round1_actions.exists():
        print(f"ERROR: work/{session_id}/ exists but is missing Round 1 artifacts "
              f"(submission_hybrid_r1.parquet and/or actions_r1.parquet). "
              f"Make sure Round 1 completed successfully for this session before running Round 2.")
        sys.exit(1)

    return session_id


def query_texts() -> dict[str, str]:
    table = pq.read_table(str(BUNDLE_DIR / "queries.parquet"), columns=["query_id", "text"])
    return dict(zip(table.column("query_id").to_pylist(), table.column("text").to_pylist()))


def train_from_round1(
    session_dir: Path,
    builder: LTRFeatureBuilder,
    bm25: BM25Retriever,
    semantic: SemanticRetriever,
) -> LambdaMartReranker:
    round1_actions = session_dir / "actions_r1.parquet"
    round1_submission = session_dir / "submission_hybrid_r1.parquet"

    actions = pd.read_parquet(round1_actions)
    submission = pd.read_parquet(round1_submission)
    served = submission.merge(
        actions[["query_id", "user_id", "item_id", "action"]],
        on=["query_id", "user_id", "item_id"],
        how="left",
    )
    served["label"] = served["action"].map(action_to_weight).fillna(0.0)
    rows: list[dict[str, Any]] = []
    texts = query_texts()
    for qid in served["query_id"].drop_duplicates():
        retrieval = builder.build_retrieval_maps(
            str(qid), bm25, semantic, top_k=TOP_CANDIDATES
        )
        for _, record in served[served["query_id"] == qid].iterrows():
            item_id = str(record["item_id"])
            item = builder.get_item_row(item_id)
            if not item:
                continue
            user_id = str(record["user_id"])
            user = builder.get_user_row(user_id)
            if not user:
                continue
            features = builder.feature_vector(
                str(qid),
                user_id,
                item_id,
                retrieval,
                texts.get(str(qid), ""),
                item,
                user,
            )
            rows.append({
                "group_id": f"{qid}::{user_id}",
                "label": int(record["label"]),
                **features,
            })
    train = pd.DataFrame(rows).dropna(subset=FEATURE_COLUMNS + ["label", "group_id"])
    if train.empty:
        raise ValueError("Round 1 action training data is empty.")
    model = LambdaMartReranker()
    model.fit(
        train[FEATURE_COLUMNS],
        train["label"].astype(int).to_numpy(),
        build_group_sizes(train),
    )
    model_path = session_dir / "ltr_round2_model.txt"
    model.save(str(model_path))
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "gain": model.model.booster_.feature_importance(importance_type="gain"),
        "split": model.model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(session_dir / "ltr_round2_feature_importance.csv", index=False)
    return model


def build_slates(
    traffic: pd.DataFrame,
    builder: LTRFeatureBuilder,
    bm25: BM25Retriever,
    semantic: SemanticRetriever,
    model: LambdaMartReranker,
) -> pd.DataFrame:
    texts = query_texts()
    rows: list[dict[str, Any]] = []
    retrieval_cache = {}
    for i, (_, impression) in enumerate(traffic.iterrows(), 1):
        qid = str(impression["query_id"])
        user_id = str(impression["user_id"])
        retrieval = retrieval_cache.get(qid)
        if retrieval is None:
            retrieval = builder.build_retrieval_maps(qid, bm25, semantic, top_k=TOP_CANDIDATES)
            retrieval_cache[qid] = retrieval
        user = builder.get_user_row(user_id)
        if not user:
            raise ValueError(f"Missing user profile for Round 2 user {user_id}.")
        candidates = retrieval["candidate_ids"]
        feature_rows = []
        valid_items = []
        for item_id in candidates:
            item = builder.get_item_row(item_id)
            if item:
                feature_rows.append(builder.feature_vector(
                    qid, user_id, item_id, retrieval, texts.get(qid, ""), item, user
                ))
                valid_items.append(item_id)
        frame = pd.DataFrame(feature_rows)
        scores = model.predict(frame[FEATURE_COLUMNS])
        ranked = [
            item_id for item_id, _ in sorted(
                zip(valid_items, scores), key=lambda pair: pair[1], reverse=True
            )
        ][:SLATE_SIZE]
        for rank, item_id in enumerate(ranked):
            rows.append({
                "world_id": WORLD_ID,
                "query_id": qid,
                "user_id": user_id,
                "rank": rank,
                "item_id": item_id,
            })
        if i % 100 == 0 or i == len(traffic):
            print(f"  Processed {i}/{len(traffic)} query/user pairs...")
    return pd.DataFrame(rows)


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


def main() -> None:
    start = time.time()

    print("[1/6] Checking simulator health...")
    health = requests.get(f"{API_BASE}/health", timeout=30)
    health.raise_for_status()
    print(f"  Simulator Healthy! World ID: {health.json().get('world_id')}")

    # Reuse Round 1's session_id — no new session is created for Round 2.
    session_id = resolve_session_id()
    session_dir = WORK_DIR / session_id
    print(f"[2/6] Reusing Round 1 session: session_id = {session_id}")
    print(f"  Local session folder: {session_dir}")

    print("[3/6] Loading retrievers and training LambdaMART on Round 1 feedback...")
    bm25 = BM25Retriever(BUNDLE_DIR)
    semantic = SemanticRetriever(BUNDLE_DIR)
    builder = LTRFeatureBuilder(BUNDLE_DIR)
    model = train_from_round1(session_dir, builder, bm25, semantic)

    # Update session.json to reflect that Round 2 now shares this session
    session_meta_path = session_dir / "session.json"
    session_meta: dict[str, Any] = {}
    if session_meta_path.exists():
        session_meta = json.loads(session_meta_path.read_text())
    session_meta.update({
        "session_id": session_id,
        "world_id": WORLD_ID,
        "rounds_used": sorted(set(session_meta.get("rounds_used", [1]) + [ROUND_INDEX])),
        "status": "round2_in_progress",
        "training_source": "round_1_actions_and_submission",
        "same_round_actions_used_for_training": False,
        "updated_at": time.time(),
    })
    session_meta_path.write_text(json.dumps(session_meta, indent=2))

    print(f"\n[4/6] Fetching Round {ROUND_INDEX} traffic for session {session_id}...")
    traffic_response = requests.get(
        f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/traffic",
        timeout=60,
    )
    traffic_response.raise_for_status()
    traffic_file = session_dir / "traffic_ltr_r2.parquet"
    traffic_file.write_bytes(traffic_response.content)
    traffic = pd.read_parquet(traffic_file).drop_duplicates(["query_id", "user_id"])
    print(f"  Downloaded traffic -> {traffic_file} ({len(traffic):,} unique pairs)")

    print("\n[5/6] Building LambdaMART slates...")
    submission = build_slates(traffic, builder, bm25, semantic, model)
    expected = len(traffic) * SLATE_SIZE
    if len(submission) != expected:
        raise ValueError(f"Expected {expected} submission rows, got {len(submission)}.")
    submission_file = session_dir / "submission_ltr_r2.parquet"
    submission.to_parquet(submission_file, index=False)
    print(f"  Saved submission -> {submission_file}")

    print(f"\n[6/6] Submitting Round {ROUND_INDEX} slates for session {session_id}...")
    with submission_file.open("rb") as handle:
        response = requests.post(
            f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/submission",
            files={"file": (submission_file.name, handle, "application/octet-stream")},
            timeout=None,
        )
    response.raise_for_status()
    result = response.json()
    print(f"  Submission accepted! Response: {result}")
    (session_dir / "submission_response_ltr_r2.json").write_text(json.dumps(result, indent=2))

    print("  Downloading actions and report...")
    actions_resp = requests.get(
        f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/actions",
        timeout=300,
    )
    if actions_resp.ok:
        actions_file = session_dir / "actions_ltr_r2.parquet"
        actions_file.write_bytes(actions_resp.content)
        print(f"  Saved actions log -> {actions_file}")
    else:
        print(f"  Actions download failed: {actions_resp.status_code} {actions_resp.text}")

    rep_resp = fetch_report_with_retry(session_id)
    report_data = None
    if rep_resp is None:
        print(f"  [Report] All retry attempts exhausted — no response received at all "
              f"(likely network/timeout). Session {session_id} left alive.")
    elif rep_resp.status_code != 200:
        print(f"  [Report] Got HTTP {rep_resp.status_code} instead of 200: {rep_resp.text[:300]}")
        print(f"  Session {session_id} left alive — report may not be ready yet.")
    else:
        report_file = session_dir / "report_ltr_r2.json"
        report_file.write_bytes(rep_resp.content)
        report_data = rep_resp.json()

    score = result.get("score", {})
    score_payload = {
        "session_id": session_id,
        "world_id": result.get("world_id", WORLD_ID),
        "round": ROUND_INDEX,
        **score,
        "baselines": result.get("baselines"),
        "deltas_vs_baselines": result.get("deltas_vs_baselines"),
    }
    (session_dir / "score_ltr_r2.json").write_text(json.dumps(score_payload, indent=2))

    (session_dir / "run_config_ltr_r2.json").write_text(json.dumps({
        "model": "LambdaMART",
        "round": ROUND_INDEX,
        "session_id": session_id,
        "training_source": "Round 1 actions joined to Round 1 submission (same session)",
        "same_round_actions_used_for_training": False,
        "top_candidates_per_channel": TOP_CANDIDATES,
        "report_saved": report_data is not None,
    }, indent=2))

    print("\n==========================================================")
    print("            ROUND 2 RESULT (LambdaMART)                   ")
    print("==========================================================")
    print(json.dumps(report_data if report_data is not None else result, indent=2))
    print("==========================================================")
    print(f"Session artifacts: {session_dir}")
    print(f"Total time: {time.time() - start:.1f}s")

    if report_data is not None:
        print(f"[Cleanup] Report received successfully — deleting session {session_id}...")
        print(f"  NOTE: this session was shared with Round 1, so this deletes BOTH "
              f"rounds' state on the simulator. Local files under work/{session_id}/ are unaffected.")
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