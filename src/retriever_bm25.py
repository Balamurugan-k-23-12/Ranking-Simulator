"""
BM25 Retriever Module
=====================
Responsible for loading the BM25 index and scoring queries against the 100,000-item catalogue.
"""

from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq

# Add bundle student directory to path to load Bm25Index
SRC_DIR    = Path(__file__).resolve().parent        # src/
ROOT_DIR   = SRC_DIR.parent                          # assignment-1/
BUNDLE_DIR = ROOT_DIR / "bundle_v1"
STUDENT_DIR = BUNDLE_DIR / "student"
if str(STUDENT_DIR) not in sys.path:
    sys.path.insert(0, str(STUDENT_DIR))

from bm25 import Bm25Index  # type: ignore


class BM25Retriever:
    def __init__(self, bundle_dir: Path = BUNDLE_DIR):
        self.bundle_dir = Path(bundle_dir)
        self.index_path = self.bundle_dir / "catalogue_bm25.pkl"
        print(f"[BM25Retriever] Loading index from {self.index_path}...")
        self.index = Bm25Index.from_pickle(self.index_path)
        
        # Load item_ids aligned with index positions
        cat_table = pq.read_table(str(self.bundle_dir / "catalogue.parquet"), columns=["item_id"])
        self.item_ids = cat_table.column("item_id").to_pylist()
        self.num_items = len(self.item_ids)
        print(f"[BM25Retriever] Loaded index with {self.num_items:,} items.")

    def score(self, query_text: str) -> np.ndarray:
        """Returns raw BM25 score array for all 100,000 items."""
        return self.index.scores(query_text)

    def get_top_k(self, query_text: str, k: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (top_k_indices, top_k_scores) sorted in descending order.
        """
        scores = self.score(query_text)
        if k >= len(scores):
            top_k_idx = np.argsort(scores)[::-1]
        else:
            # Fast partial sort
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        return top_k_idx, scores[top_k_idx]


if __name__ == "__main__":
    # Self-test
    retriever = BM25Retriever()
    test_query = "usb c laptop charger"
    top_indices, top_scores = retriever.get_top_k(test_query, k=5)
    print(f"\nTop 5 BM25 matches for '{test_query}':")
    for rank, (idx, sc) in enumerate(zip(top_indices, top_scores), start=1):
        print(f"  Rank {rank}: Item {retriever.item_ids[idx]} (Score: {sc:.4f})")
