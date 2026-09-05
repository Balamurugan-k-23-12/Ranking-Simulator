"""
Semantic Retriever Module
=========================
Responsible for dense vector similarity scoring using pre-computed embeddings
for the 100,000 catalogue items and 144 queries.
"""

from pathlib import Path
import warnings
import numpy as np
import pyarrow.parquet as pq

# Suppress spurious OpenBLAS / Accelerate matmul warnings on Apple Silicon
warnings.filterwarnings("ignore", category=RuntimeWarning)

SRC_DIR    = Path(__file__).resolve().parent        # src/
ROOT_DIR   = SRC_DIR.parent                          # assignment-1/
BUNDLE_DIR = ROOT_DIR / "bundle_v1"


class SemanticRetriever:
    def __init__(self, bundle_dir: Path = BUNDLE_DIR):
        self.bundle_dir = Path(bundle_dir)
        print("[SemanticRetriever] Loading item embeddings from catalogue.parquet...")
        cat_table = pq.read_table(str(self.bundle_dir / "catalogue.parquet"), columns=["item_id", "embedding"])
        self.item_ids = cat_table.column("item_id").to_pylist()
        self.num_items = len(self.item_ids)

        self.item_matrix = np.asarray(cat_table.column("embedding").to_pylist(), dtype=np.float64)
        item_norms = np.linalg.norm(self.item_matrix, axis=1, keepdims=True)
        self.item_matrix = self.item_matrix / np.where(item_norms > 0, item_norms, 1.0)
        print(f"[SemanticRetriever] Loaded {self.num_items:,} item embeddings with shape {self.item_matrix.shape}.")

        # Load query embeddings
        print("[SemanticRetriever] Loading query embeddings from query_embeddings.parquet...")
        qe_table = pq.read_table(str(self.bundle_dir / "query_embeddings.parquet"))
        qe_qids = qe_table.column("query_id").to_pylist()
        raw_qe_vecs = np.asarray(qe_table.column("embedding").to_pylist(), dtype=np.float64)
        q_norms = np.linalg.norm(raw_qe_vecs, axis=1, keepdims=True)
        normalized_qe_vecs = raw_qe_vecs / np.where(q_norms > 0, q_norms, 1.0)

        self.query_vectors = {qid: normalized_qe_vecs[i] for i, qid in enumerate(qe_qids)}
        print(f"[SemanticRetriever] Loaded {len(self.query_vectors)} query embeddings.")

    def score(self, query_id: str) -> np.ndarray:
        """Returns cosine similarity scores for all 100,000 items."""
        q_vec = self.query_vectors.get(query_id)
        if q_vec is None:
            return np.zeros(self.num_items, dtype=np.float32)
        return np.dot(self.item_matrix, q_vec)

    def get_top_k(self, query_id: str, k: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (top_k_indices, top_k_scores) sorted in descending order.
        """
        scores = self.score(query_id)
        if k >= len(scores):
            top_k_idx = np.argsort(scores)[::-1]
        else:
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        return top_k_idx, scores[top_k_idx]


if __name__ == "__main__":
    # Self-test
    retriever = SemanticRetriever()
    test_qid = "Q00000"
    top_indices, top_scores = retriever.get_top_k(test_qid, k=5)
    print(f"\nTop 5 Semantic matches for query_id '{test_qid}':")
    for rank, (idx, sc) in enumerate(zip(top_indices, top_scores), start=1):
        print(f"  Rank {rank}: Item {retriever.item_ids[idx]} (Cosine Sim: {sc:.4f})")
