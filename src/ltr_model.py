"""LightGBM LambdaMART wrapper for the Round 1 / Round 2 reranking task."""

from __future__ import annotations

from typing import List, Sequence

import lightgbm as lgb
import pandas as pd


class LambdaMartReranker:
    """Simple LambdaMART wrapper using LightGBM LGBMRanker."""

    def __init__(self, params: dict | None = None):
        default_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": [20],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 8,
            "min_data_in_leaf": 30,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "verbosity": -1,
            "n_estimators": 150,
        }
        if params:
            default_params.update(params)
        self.model = lgb.LGBMRanker(**default_params)

    def fit(self, X: pd.DataFrame, y: Sequence[float], groups: Sequence[int]) -> "LambdaMartReranker":
        self.model.fit(X, y, group=groups)
        return self

    def predict(self, X: pd.DataFrame) -> List[float]:
        return self.model.predict(X)

    def save(self, model_path: str) -> None:
        self.model.booster_.save_model(model_path)

    @classmethod
    def load(cls, model_path: str) -> "LambdaMartReranker":
        ranker = cls()
        ranker.model = lgb.Booster(model_file=model_path)
        return ranker


def build_group_sizes(df: pd.DataFrame, group_col: str = "group_id") -> List[int]:
    """
    Computes group sizes in the EXACT insertion order of rows in df.
    Using sort=False ensures group boundaries are not alphabetically scrambled!
    """
    return df.groupby(group_col, sort=False).size().tolist()
