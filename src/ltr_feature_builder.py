"""Feature construction utilities for a Round 1 LambdaMART reranker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
BUNDLE_DIR = ROOT_DIR / "bundle_v1"
RESULTS_DIR = ROOT_DIR / "results" / "round_1"


def load_catalogue(bundle_dir: Path = BUNDLE_DIR) -> pd.DataFrame:
    table = pq.read_table(str(bundle_dir / "catalogue.parquet"))
    return table.to_pandas()


def load_queries(bundle_dir: Path = BUNDLE_DIR) -> pd.DataFrame:
    table = pq.read_table(str(bundle_dir / "queries.parquet"))
    return table.to_pandas()


def load_users(bundle_dir: Path = BUNDLE_DIR) -> pd.DataFrame:
    table = pq.read_table(str(bundle_dir / "users.parquet"))
    return table.to_pandas()


def load_actions(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    table = pq.read_table(str(results_dir / "actions_r1.parquet"))
    return table.to_pandas()


def load_traffic(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    table = pq.read_table(str(results_dir / "traffic_r1.parquet"))
    return table.to_pandas()


def action_to_weight(action: str) -> float:
    if not isinstance(action, str):
        return 0.0
    action = action.lower()
    weights = {
        "seen": 0.0,
        "click": 1.0,
        "cart": 3.0,
        "purchase": 10.0,
    }
    return weights.get(action, 0.0)


def normalize_price(price: float, median_price: float) -> float:
    if median_price <= 0:
        return 0.0
    return float(price) / float(median_price)


def compute_quality_score(price: float, rating: float) -> float:
    rating_score = float(np.clip(rating / 5.0, 0.0, 1.0))
    price_factor = 1.0 / (1.0 + np.log1p(max(0.0, price - 100.0) / 100.0))
    return float(rating_score * price_factor)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class LTRFeatureBuilder:
    """Create query-user-item features and training rows for Round 1 LambdaMART."""

    def __init__(self, bundle_dir: Path = BUNDLE_DIR):
        self.bundle_dir = Path(bundle_dir)
        self.catalogue = load_catalogue(bundle_dir)
        self.users = load_users(bundle_dir)
        self.queries = load_queries(bundle_dir)
        self.item_by_id = self.catalogue.set_index("item_id").to_dict("index")
        self.user_by_id = self.users.set_index("user_id").to_dict("index")
        self.query_text = dict(zip(self.queries["query_id"], self.queries["text"]))
        self.median_price = float(self.catalogue["price"].median())
        self.category_median_price = self.catalogue.groupby("category")["price"].median().to_dict()
        self.category_mean_rating = self.catalogue.groupby("category")["rating"].mean().to_dict()

    def get_query_text(self, query_id: str) -> str:
        return str(self.query_text.get(query_id, ""))

    def get_item_row(self, item_id: str) -> Dict[str, Any]:
        return self.item_by_id.get(item_id, {})

    def get_user_row(self, user_id: str) -> Dict[str, Any]:
        return self.user_by_id.get(user_id, {})

    def _token_set(self, text: str) -> set[str]:
        return {tok.lower() for tok in str(text).split() if tok}

    def build_retrieval_maps(
        self,
        query_id: str,
        bm25_retriever: Any,
        sem_retriever: Any,
        top_k: int = 2000,
    ) -> Dict[str, Any]:
        q_text = self.get_query_text(query_id)
        bm25_idx, bm25_scores = bm25_retriever.get_top_k(q_text, k=top_k)
        sem_idx, sem_scores = sem_retriever.get_top_k(query_id, k=top_k)

        bm25_score_map: Dict[str, float] = {}
        bm25_rank_map: Dict[str, int] = {}
        for rank, idx in enumerate(bm25_idx, start=1):
            item_id = bm25_retriever.item_ids[int(idx)]
            bm25_score_map[item_id] = float(bm25_scores[rank - 1])
            bm25_rank_map[item_id] = int(rank)

        sem_score_map: Dict[str, float] = {}
        sem_rank_map: Dict[str, int] = {}
        for rank, idx in enumerate(sem_idx, start=1):
            item_id = sem_retriever.item_ids[int(idx)]
            sem_score_map[item_id] = float(sem_scores[rank - 1])
            sem_rank_map[item_id] = int(rank)

        candidate_ids = sorted(
            set(bm25_score_map) | set(sem_score_map),
            key=lambda item_id: (
                max(bm25_score_map.get(item_id, -1e18), sem_score_map.get(item_id, -1e18)),
                -bm25_rank_map.get(item_id, 10**9),
                -sem_rank_map.get(item_id, 10**9),
            ),
            reverse=True,
        )

        return {
            "candidate_ids": candidate_ids,
            "bm25_score": bm25_score_map,
            "bm25_rank": bm25_rank_map,
            "semantic_score": sem_score_map,
            "semantic_rank": sem_rank_map,
        }

    def feature_vector(
        self,
        query_id: str,
        user_id: str,
        item_id: str,
        retrieval_map: Dict[str, Any],
        q_text: str,
        item_row: Dict[str, Any],
        user_row: Dict[str, Any],
    ) -> Dict[str, float]:
        item_id = str(item_id)
        q_tokens = self._token_set(q_text)
        title = str(item_row.get("title", ""))
        description = str(item_row.get("description", ""))
        title_tokens = self._token_set(title)
        desc_tokens = self._token_set(description)

        price = safe_float(item_row.get("price", 0.0))
        rating = safe_float(item_row.get("rating", 0.0))
        quality_score = compute_quality_score(price, rating)
        category = str(item_row.get("category", ""))
        brand = str(item_row.get("brand", ""))

        bm25_score = safe_float(retrieval_map["bm25_score"].get(item_id, 0.0))
        bm25_rank = safe_float(retrieval_map["bm25_rank"].get(item_id, 1e9))
        semantic_score = safe_float(retrieval_map["semantic_score"].get(item_id, 0.0))
        semantic_rank = safe_float(retrieval_map["semantic_rank"].get(item_id, 1e9))

        category_match = 1.0 if category.lower() in {tok.lower() for tok in q_tokens} else 0.0
        brand_match = 1.0 if brand.lower() in {tok.lower() for tok in q_tokens} else 0.0
        title_overlap = len(q_tokens & title_tokens)
        description_overlap = len(q_tokens & desc_tokens)
        category_median = safe_float(
            self.category_median_price.get(category, self.median_price),
            self.median_price,
        )
        category_mean_rating = safe_float(
            self.category_mean_rating.get(category, self.catalogue["rating"].mean()),
            float(self.catalogue["rating"].mean()),
        )
        cheap_item_flag = 1.0 if price <= category_median else 0.0
        log_price = float(np.log1p(max(price, 0.0)))
        relative_price = float(np.log1p(max(price, 0.0)) - np.log1p(max(category_median, 0.0)))
        rating_norm = float(np.clip(rating / 5.0, 0.0, 1.0))
        rating_vs_category = float(rating - category_mean_rating)
        query_phrase = " ".join(str(q_text).lower().split())
        title_normalized = " ".join(title.lower().split())
        phrase_match = 1.0 if query_phrase and query_phrase in title_normalized else 0.0
        semantic_margin = float(semantic_score - np.mean(list(retrieval_map["semantic_score"].values())))
        semantic_rank_inverse = float(1.0 / (1.0 + semantic_rank))
        bm25_rank_inverse = float(1.0 / (1.0 + bm25_rank))
        user_price_sensitivity = safe_float(user_row.get("price_sensitivity"))
        user_brand_loyalty = safe_float(user_row.get("brand_loyalty"))
        user_review_dependency = safe_float(user_row.get("review_dependency"))
        user_ad_susceptibility = safe_float(user_row.get("ad_susceptibility"))
        user_domain_knowledge = safe_float(user_row.get("domain_knowledge"))
        user_novelty_preference = safe_float(user_row.get("novelty_preference"))
        user_variety_seeking = safe_float(user_row.get("variety_seeking"))
        user_deal_seeking = safe_float(user_row.get("deal_seeking"))
        segment = str(user_row.get("segment", "")).lower()

        return {
            "bm25_score": float(bm25_score),
            "bm25_rank": float(bm25_rank),
            "semantic_score": float(semantic_score),
            "semantic_rank": float(semantic_rank),
            "quality_score": float(quality_score),
            "price": float(price),
            "rating": float(rating),
            "price_norm": normalize_price(price, self.median_price),
            "log_price": log_price,
            "relative_price": relative_price,
            "rating_norm": rating_norm,
            "rating_vs_category": rating_vs_category,
            "semantic_margin": semantic_margin,
            "semantic_rank_inverse": semantic_rank_inverse,
            "bm25_rank_inverse": bm25_rank_inverse,
            "category_match": float(category_match),
            "brand_match": float(brand_match),
            "title_overlap_count": float(title_overlap),
            "description_overlap_count": float(description_overlap),
            "title_overlap_ratio": float(title_overlap / max(len(q_tokens), 1)),
            "description_overlap_ratio": float(description_overlap / max(len(q_tokens), 1)),
            "exact_phrase_match": phrase_match,
            "cheap_item_flag": float(cheap_item_flag),
            "user_price_sensitivity": user_price_sensitivity,
            "user_brand_loyalty": user_brand_loyalty,
            "user_review_dependency": user_review_dependency,
            "user_ad_susceptibility": user_ad_susceptibility,
            "user_domain_knowledge": user_domain_knowledge,
            "user_novelty_preference": user_novelty_preference,
            "user_variety_seeking": user_variety_seeking,
            "user_deal_seeking": user_deal_seeking,
            "segment_bargain": float(segment == "bargain"),
            "segment_brand_loyal": float(segment == "brand_loyal"),
            "segment_general": float(segment == "general"),
            "segment_gifter": float(segment == "gifter"),
            "segment_impulsive": float(segment == "impulsive"),
            "segment_minimalist": float(segment == "minimalist"),
            "segment_premium": float(segment == "premium"),
            "segment_techie": float(segment == "techie"),
            "user_price_affinity": float(user_price_sensitivity * relative_price),
            "user_brand_affinity": float(user_brand_loyalty * brand_match),
            "user_review_affinity": float(user_review_dependency * rating_norm),
        }

    def build_training_rows(
        self,
        actions_df: pd.DataFrame,
        bm25_retriever: Any,
        sem_retriever: Any,
        top_k: int = 2000,
        max_queries: int | None = None,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        unique_query_ids = list(actions_df["query_id"].dropna().unique())
        if max_queries is not None:
            unique_query_ids = unique_query_ids[:max_queries]

        for query_id in unique_query_ids:
            retrieval_map = self.build_retrieval_maps(query_id, bm25_retriever, sem_retriever, top_k=top_k)
            candidate_ids = retrieval_map["candidate_ids"]
            q_text = self.get_query_text(query_id)

            user_item_reward = (
                actions_df[(actions_df["query_id"] == query_id)]
                .groupby(["user_id", "item_id"], as_index=False)["action"]
                .agg(lambda vals: max(action_to_weight(str(v)) for v in vals))
                .rename(columns={"action": "label"})
            )

            for _, row in user_item_reward.iterrows():
                user_id = str(row["user_id"])
                item_id = str(row["item_id"])
                label = float(row["label"])

                item_row = self.get_item_row(item_id)
                user_row = self.get_user_row(user_id)
                if not item_row or not user_row:
                    continue

                feat = self.feature_vector(query_id, user_id, item_id, retrieval_map, q_text, item_row, user_row)
                feat_row = {
                    "query_id": query_id,
                    "user_id": user_id,
                    "item_id": item_id,
                    "group_id": f"{query_id}::{user_id}",
                    "label": label,
                    **feat,
                }
                rows.append(feat_row)

            observed_items = set(user_item_reward["item_id"].tolist())
            for candidate_id in candidate_ids:
                if candidate_id in observed_items:
                    continue
                for user_id in actions_df[actions_df["query_id"] == query_id]["user_id"].dropna().unique():
                    item_row = self.get_item_row(candidate_id)
                    user_row = self.get_user_row(str(user_id))
                    if not item_row or not user_row:
                        continue
                    feat = self.feature_vector(query_id, str(user_id), candidate_id, retrieval_map, q_text, item_row, user_row)
                    rows.append(
                        {
                            "query_id": query_id,
                            "user_id": str(user_id),
                            "item_id": candidate_id,
                            "group_id": f"{query_id}::{user_id}",
                            "label": 0.0,
                            **feat,
                        }
                    )

        return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "bm25_score",
    "bm25_rank",
    "semantic_score",
    "semantic_rank",
    "quality_score",
    "price",
    "rating",
    "price_norm",
    "log_price",
    "relative_price",
    "rating_norm",
    "rating_vs_category",
    "semantic_margin",
    "semantic_rank_inverse",
    "bm25_rank_inverse",
    "category_match",
    "brand_match",
    "title_overlap_count",
    "description_overlap_count",
    "title_overlap_ratio",
    "description_overlap_ratio",
    "exact_phrase_match",
    "cheap_item_flag",
    "user_price_sensitivity",
    "user_brand_loyalty",
    "user_review_dependency",
    "user_ad_susceptibility",
    "user_domain_knowledge",
    "user_novelty_preference",
    "user_variety_seeking",
    "user_deal_seeking",
    "segment_bargain",
    "segment_brand_loyal",
    "segment_general",
    "segment_gifter",
    "segment_impulsive",
    "segment_minimalist",
    "segment_premium",
    "segment_techie",
    "user_price_affinity",
    "user_brand_affinity",
    "user_review_affinity",
]
