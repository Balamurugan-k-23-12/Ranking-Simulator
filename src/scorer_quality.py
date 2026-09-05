"""
Quality Scorer Module
=====================
Responsible for computing product quality and popularity signals
based on catalogue metadata (ratings, price).
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence, Union
import numpy as np
import pyarrow.parquet as pq

SRC_DIR    = Path(__file__).resolve().parent        # src/
ROOT_DIR   = SRC_DIR.parent                          # assignment-1/
BUNDLE_DIR = ROOT_DIR / "bundle_v1"


class QualityScorer:
    def __init__(self, bundle_dir: Path = BUNDLE_DIR):
        self.bundle_dir = Path(bundle_dir)
        print("[QualityScorer] Loading catalogue metadata...")
        cat_table = pq.read_table(
            str(self.bundle_dir / "catalogue.parquet"),
            columns=["item_id", "rating", "price", "category", "brand"]
        )
        self.item_ids = cat_table.column("item_id").to_pylist()
        self.ratings = np.asarray(cat_table.column("rating").to_pylist(), dtype=np.float32)
        self.prices = np.asarray(cat_table.column("price").to_pylist(), dtype=np.float32)
        self.num_items = len(self.item_ids)

        # Composite static quality score:
        # 1. Rating score (0.0 to 1.0)
        norm_rating = np.clip(self.ratings / 5.0, 0.0, 1.0)

        # 2. Moderate price reasonableness penalty for extreme outliers
        price_factor = 1.0 / (1.0 + np.log1p(np.maximum(0.0, self.prices - 100.0) / 100.0))

        self.quality_scores = norm_rating * price_factor
        print(f"[QualityScorer] Computed static quality scores for {self.num_items:,} items.")

    def get_quality_ranks_for_candidates(self, candidate_indices: Union[np.ndarray, list[int]]) -> dict[int, int]:
        """
        Given a set of candidate item indices, ranks them among themselves by quality score (1 = best).
        Returns a mapping from item_idx -> quality_rank (1-based).
        """
        cand_arr = np.asarray(candidate_indices, dtype=np.int64)
        cand_scores = self.quality_scores[cand_arr]
        
        # Sort descending
        sorted_order = np.argsort(cand_scores)[::-1]
        
        # Map item_idx -> rank (1-based)
        quality_ranks = {}
        for rank_zero_based, orig_pos in enumerate(sorted_order):
            item_idx = int(cand_arr[orig_pos])
            quality_ranks[item_idx] = rank_zero_based + 1
        return quality_ranks


if __name__ == "__main__":
    # Self-test
    scorer = QualityScorer()
    sample_candidates = [0, 1, 2, 3, 4]
    ranks = scorer.get_quality_ranks_for_candidates(sample_candidates)
    print("\nQuality ranking for sample items:")
    for idx in sample_candidates:
        print(f"  Item {scorer.item_ids[idx]}: Rating={scorer.ratings[idx]}, Price=${scorer.prices[idx]:.2f}, Rank={ranks[idx]}")
