"""Round 1 LambdaMART submission pipeline.

This is a separate simulator-facing submission script, deliberately isolated from
hybrid_rrf_r1.py. It trains or loads a LambdaMART reranker on the available
Round 1 data, generates BM25 + semantic candidate pools, reranks them by learned
score, and submits the final Top-20 slate to the simulator.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import requests

from ltr_feature_builder import FEATURE_COLUMNS, LTRFeatureBuilder
from ltr_model import LambdaMartReranker, build_group_sizes
from retriever_bm25 import BM25Retriever
from retriever_semantic import SemanticRetriever

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
BUNDLE_DIR = ROOT_DIR / "bundle_v1"
RESULTS_DIR = ROOT_DIR / "results" / "round_1"
WORK_DIR = ROOT_DIR / "work"
MODEL_PATH = RESULTS_DIR / "ltr_query_item_reranker_model.txt"
TRAFFIC_FILE = RESULTS_DIR / "traffic_ltr_r1.parquet"
SUBMISSION_FILE = RESULTS_DIR / "submission_ltr_r1.parquet"
ACTIONS_FILE = RESULTS_DIR / "actions_ltr_r1.parquet"
REPORT_FILE = RESULTS_DIR / "report_ltr_r1.json"

API_BASE = os.getenv("SIMULATOR_API_BASE", "http://127.0.0.1:8000")
WORLD_ID = "w908832361965e48e"
SLATE_SIZE = 20
ROUND_INDEX = 1
TOP_CANDIDATES = 2000

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)


def load_query_texts(bundle_dir: Path = BUNDLE_DIR) -> dict[str, str]:
    q_table = pq.read_table(str(bundle_dir / "queries.parquet"), columns=["query_id", "text"])
    qids = q_table.column("query_id").to_pylist()
    texts = q_table.column("text").to_pylist()
    return dict(zip(qids, texts))


def ensure_model(
    bm25_retriever: BM25Retriever,
    sem_retriever: SemanticRetriever,
    bundle_dir: Path = BUNDLE_DIR,
    force_retrain: bool = False,
) -> LambdaMartReranker:
    if MODEL_PATH.exists() and not force_retrain:
        print(f"[LTR] Loading cached model from {MODEL_PATH}...")
        return LambdaMartReranker.load(str(MODEL_PATH))

    print(f"[LTR] Training LambdaMART model with top-{TOP_CANDIDATES} candidates...")
    builder = LTRFeatureBuilder(bundle_dir=bundle_dir)
    query_ids = list(builder.query_text)
    rows: list[dict[str, Any]] = []
    
    for query_id in query_ids:
        retrieval_map = builder.build_retrieval_maps(
            query_id, bm25_retriever, sem_retriever, top_k=TOP_CANDIDATES
        )
        candidate_ids = retrieval_map["candidate_ids"]
        q_text = builder.get_query_text(query_id)
        for item_id in candidate_ids:
            item_row = builder.get_item_row(item_id)
            if not item_row:
                continue
            features = builder.feature_vector(
                query_id,
                "proxy-user",
                item_id,
                retrieval_map,
                q_text,
                item_row,
                {},
            )
            sem_rank = retrieval_map["semantic_rank"].get(item_id, TOP_CANDIDATES + 1)
            bm25_rank = retrieval_map["bm25_rank"].get(item_id, TOP_CANDIDATES + 1)
            # Pre-submission proxy target
            label = (
                2.0 / (1.0 + float(sem_rank))
                + 0.5 / (1.0 + float(bm25_rank))
                + 0.05 * features["quality_score"]
            )
            rows.append(
                {
                    "group_id": query_id,
                    "label": label,
                    **features,
                }
            )

    train_df = pd.DataFrame(rows)
    if train_df.empty:
        raise ValueError("Proxy training data is empty. Check bundle and retrieval inputs.")
        
    # LambdaMART expects integer relevance grades (4=best, 0=least relevant)
    train_df["label"] = (
        train_df.groupby("group_id", sort=False)["label"]
        .rank(method="first", ascending=False)
        .apply(lambda rank: 4 if rank <= 10 else 3 if rank <= 50 else 2 if rank <= 200 else 1 if rank <= 1000 else 0)
        .astype(int)
    )
    
    max_group_rows = 9500
    grouped_rows: list[pd.DataFrame] = []
    for _, group_frame in train_df.groupby("group_id", sort=False):
        if len(group_frame) > max_group_rows:
            positive = group_frame[group_frame["label"] > 0]
            negatives = group_frame[group_frame["label"] == 0].head(
                max_group_rows - len(positive)
            )
            group_frame = pd.concat([positive, negatives], ignore_index=True)
        grouped_rows.append(group_frame)
    train_df = pd.concat(grouped_rows, ignore_index=True)
    
    print(
        "[LTR] Proxy grades distribution:",
        train_df["label"].value_counts().sort_index().to_dict(),
    )

    X = train_df[FEATURE_COLUMNS]
    y = train_df["label"].astype(float).to_numpy()
    groups = build_group_sizes(train_df, group_col="group_id")

    model = LambdaMartReranker()
    model.fit(X, y, groups)
    model.save(str(MODEL_PATH))
    
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "gain": model.model.booster_.feature_importance(importance_type="gain"),
        "split": model.model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(RESULTS_DIR / "ltr_feature_importance.csv", index=False)
    return model


def score_query_candidates(
    model: LambdaMartReranker,
    builder: LTRFeatureBuilder,
    qid: str,
    candidate_ids: list[str],
    retrieval_map: dict[str, Any],
    query_text: str,
) -> list[str]:
    rows: list[dict[str, Any]] = []
    for item_id in candidate_ids:
        item_row = builder.get_item_row(item_id)
        if not item_row:
            continue
        feature_map = builder.feature_vector(qid, "proxy-user", item_id, retrieval_map, query_text, item_row, {})
        rows.append({
            "query_id": qid,
            "item_id": item_id,
            **feature_map,
        })

    if not rows:
        return candidate_ids[:SLATE_SIZE]

    frame = pd.DataFrame(rows)
    for c in FEATURE_COLUMNS:
        if c not in frame.columns:
            frame[c] = 0.0

    scores = model.predict(frame[FEATURE_COLUMNS])
    ranked = sorted(zip(frame["item_id"].tolist(), scores.tolist()), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in ranked[:SLATE_SIZE]]


def build_ltr_slates(
    traffic_df: pd.DataFrame,
    bm25_retriever: BM25Retriever,
    sem_retriever: SemanticRetriever,
    builder: LTRFeatureBuilder,
    query_texts: dict[str, str],
    model: LambdaMartReranker,
) -> pd.DataFrame:
    unique_queries = traffic_df["query_id"].unique()
    print(f"\n[LTR Submission] Ranking {len(unique_queries)} unique queries...")

    # Fast caching: compute top 20 items per query_id ONCE
    query_top20_items: dict[str, list[str]] = {}
    for i, qid in enumerate(unique_queries, start=1):
        q_text = query_texts.get(qid, "")
        retrieval_map = builder.build_retrieval_maps(qid, bm25_retriever, sem_retriever, top_k=TOP_CANDIDATES)
        candidate_ids = retrieval_map["candidate_ids"]
        
        query_top20_items[qid] = score_query_candidates(
            model=model,
            builder=builder,
            qid=qid,
            candidate_ids=candidate_ids,
            retrieval_map=retrieval_map,
            query_text=q_text,
        )

        if i % 30 == 0 or i == len(unique_queries):
            print(f"  Processed {i}/{len(unique_queries)} queries...")

    # Expand to all traffic rows
    rows: list[dict[str, Any]] = []
    for _, tr in traffic_df.iterrows():
        qid = str(tr["query_id"])
        uid = str(tr["user_id"])
        slate = query_top20_items.get(qid, [])
        for rank, item_id in enumerate(slate[:SLATE_SIZE]):
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
    print(f"  ✓ Row count verified: {len(submission_df):,} rows.")
    print(f"  ✓ Schema and non-null verified.")


def main() -> None:
    print("==========================================================")
    print("      ROUND 1: LAMBDA MART RERANKER SUBMISSION            ")
    print("==========================================================")
    start_time = time.time()

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

    print("\n[2/6] Pre-loading retrieval and LTR modules...")
    bm25_retriever = BM25Retriever(BUNDLE_DIR)
    sem_retriever = SemanticRetriever(BUNDLE_DIR)
    builder = LTRFeatureBuilder(BUNDLE_DIR)
    query_texts = load_query_texts(BUNDLE_DIR)
    model = ensure_model(bm25_retriever, sem_retriever, bundle_dir=BUNDLE_DIR, force_retrain=True)
    print(f"  Modules ready. Model path: {MODEL_PATH}")

    print(f"\n[3/6] Opening simulator session...")
    sess_resp = requests.post(f"{API_BASE}/v1/sessions?seed=42", timeout=30)
    if sess_resp.status_code != 200:
        print(f"  Failed to create session: {sess_resp.text}")
        return
    session_id = sess_resp.json()["session_id"]
    session_dir = WORK_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Session started: session_id = {session_id}")

    print(f"\n[4/6] Fetching Round {ROUND_INDEX} traffic...")
    traf_resp = requests.get(f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/traffic", timeout=60)
    if traf_resp.status_code == 200:
        with open(TRAFFIC_FILE, "wb") as f:
            f.write(traf_resp.content)
        (session_dir / "traffic_ltr_r1.parquet").write_bytes(traf_resp.content)

    traffic_raw = pd.read_parquet(TRAFFIC_FILE)
    traffic = traffic_raw.drop_duplicates(subset=["query_id", "user_id"]).reset_index(drop=True)
    print(f"  Unique (query_id, user_id) pairs to rank: {len(traffic):,}")

    print("\n[5/6] Building LTR slates...")
    submission = build_ltr_slates(
        traffic_df=traffic,
        bm25_retriever=bm25_retriever,
        sem_retriever=sem_retriever,
        builder=builder,
        query_texts=query_texts,
        model=model,
    )
    validate_submission(submission, len(traffic))
    submission.to_parquet(SUBMISSION_FILE, index=False)
    (session_dir / "submission_ltr_r1.parquet").write_bytes(SUBMISSION_FILE.read_bytes())
    print(f"  Saved submission at {SUBMISSION_FILE}")

    print(f"\n[6/6] Submitting Round {ROUND_INDEX} slate to simulator...")
    with open(SUBMISSION_FILE, "rb") as f:
        up_resp = requests.post(
            f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/submission",
            files={"file": (SUBMISSION_FILE.name, f, "application/octet-stream")},
            timeout=None,
        )
    if up_resp.status_code != 200:
        print(f"  Submission failed ({up_resp.status_code}): {up_resp.text}")
        return
    print(f"  Submission accepted! Response: {up_resp.json()}")
    submission_response = up_resp.json()
    (session_dir / "submission_response.json").write_text(json.dumps(submission_response, indent=2))

    act_resp = requests.get(f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/actions", timeout=600)
    if act_resp.status_code == 200:
        with open(ACTIONS_FILE, "wb") as f:
            f.write(act_resp.content)
        (session_dir / "actions_ltr_r1.parquet").write_bytes(act_resp.content)
        print(f"  Saved actions: {ACTIONS_FILE} ({len(act_resp.content)/1024:.1f} KB)")

    rep_resp = requests.get(f"{API_BASE}/v1/sessions/{session_id}/rounds/{ROUND_INDEX}/report", timeout=600)
    if rep_resp.status_code == 200:
        with open(REPORT_FILE, "wb") as f:
            f.write(rep_resp.content)
        print(f"  Saved report: {REPORT_FILE}")
        report_data = rep_resp.json()
        print("\n==========================================================")
        print("            OFFICIAL LTR ROUND 1 EVALUATION REPORT          ")
        print("==========================================================")
        print(json.dumps(report_data, indent=2))

    print(f"\nTotal time elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
