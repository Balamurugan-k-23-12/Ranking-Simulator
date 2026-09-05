"""Round 1 LambdaMART reranker pipeline.

This module builds a candidate pool from BM25 and semantic retrieval, creates
query-user-item features, trains a LightGBM LambdaMART model, and exports the
trained model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ltr_feature_builder import (
    FEATURE_COLUMNS,
    LTRFeatureBuilder,
    load_actions,
    load_traffic,
)
from ltr_model import LambdaMartReranker, build_group_sizes
from retriever_bm25 import BM25Retriever
from retriever_semantic import SemanticRetriever

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
BUNDLE_DIR = ROOT_DIR / "bundle_v1"
RESULTS_DIR = ROOT_DIR / "results" / "round_1"
MODEL_PATH = RESULTS_DIR / "ltr_reranker_model.txt"


def build_training_data(
    bundle_dir: Path = BUNDLE_DIR,
    results_dir: Path = RESULTS_DIR,
    top_k: int = 2000,
    max_queries: int | None = None,
) -> pd.DataFrame:
    feature_builder = LTRFeatureBuilder(bundle_dir=bundle_dir)
    bm25_retriever = BM25Retriever(bundle_dir=bundle_dir)
    sem_retriever = SemanticRetriever(bundle_dir=bundle_dir)
    actions_df = load_actions(results_dir)

    if max_queries is not None:
        query_ids = list(actions_df["query_id"].dropna().unique())[:max_queries]
        actions_df = actions_df[actions_df["query_id"].isin(query_ids)].copy()

    return feature_builder.build_training_rows(actions_df, bm25_retriever, sem_retriever, top_k=top_k, max_queries=max_queries)


def train_lambda_mart(
    train_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[LambdaMartReranker, pd.DataFrame]:
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    train_df = train_df.dropna(subset=feature_columns + ["label", "group_id"]).copy()
    X = train_df[feature_columns]
    y = train_df["label"].astype(float).to_numpy()
    groups = build_group_sizes(train_df, group_col="group_id")

    model = LambdaMartReranker()
    model.fit(X, y, groups)
    return model, train_df


def main() -> None:
    parser = argparse.ArgumentParser(description="LambdaMART reranker for Round 1.")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print("[LTR] Building candidate pool and feature rows...")
    train_df = build_training_data(
        bundle_dir=BUNDLE_DIR,
        results_dir=RESULTS_DIR,
        top_k=args.top_k,
        max_queries=args.max_queries if not args.smoke_test else min(args.max_queries or 5, 5),
    )

    if train_df.empty:
        raise ValueError("Training frame is empty. Check the retrieval/data inputs.")

    print(f"[LTR] Training rows: {len(train_df):,}")
    model, _ = train_lambda_mart(train_df, feature_columns=FEATURE_COLUMNS)
    model.save(args.model_path)
    print(f"[LTR] Saved model to {args.model_path}")


if __name__ == "__main__":
    main()
