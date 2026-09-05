# Information Retrieval (IRE) Learning-to-Rank Project
## Experiment Log & Architecture Documentation

### 1. Environment & World Specification
- **World ID**: `w908832361965e48e`
- **Catalogue Size**: 100,000 electronics items
- **Universe**: 400 users, 144 queries
- **Slate Size**: 20 ranked items per impression (Rank 0 = top, Rank 19 = bottom)
- **Traffic per Round**: 14,400 search impressions (14,216 unique query-user pairs)
- **Submission Size**: 284,320 rows ($14,216 \times 20$)
- **Action Levels**: `seen` (0) → `click` (1) → `cart` (2) → `purchase` (3)

---

### 2. Experiment Roadmap & Results Log

| Run # | Strategy / Model | Round | Expected Value | Sampled Value | CTR (%) | Cart Rate (%) | Purchase Rate (%) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **00 (Base)** | Random Baseline | - | 0.03439 | - | - | - | - | Provided by simulator |
| **00 (Base)** | Semantic Baseline | - | 0.07932 | - | - | - | - | Dense vector similarity |
| **01 (Sanity)** | Pure BM25 Baseline | R1 | **0.07198** | **0.07103** | **8.09%** | **7.19%** | **4.14%** | Pure lexical search using `catalogue_bm25.pkl` |
| **02 (R1 Best)**| 3-Way Reciprocal Rank Fusion (RRF) | R1 | **0.07310** | **0.07208** | **8.14%** | **7.23%** | **4.22%** | BM25+Semantic+Quality, RRF k=60, w_qual=0.2. Beats BM25 +3.1%, below Semantic -7.8% |
| **03 (R2)** | Supervised LTR (LightGBM) | R2 | *Upcoming* | - | - | - | - | Trained on Round 1 action logs |
| **04 (R3)** | Debiased LTR (IPW + Multi-task) | R3 | *Upcoming* | - | - | - | - | Trained on R1+R2 cumulative logs with position debiasing |

---

### 3. Modular System Architecture (Round 1)

The Round 1 submission is organized into independent, modular components:

1. **`retriever_bm25.py`** (`BM25Retriever`):
   - Loads pre-indexed Okapi BM25 (`bundle_v1/catalogue_bm25.pkl`).
   - Retrieves top-$K$ lexical/keyword candidates for each query text.

2. **`retriever_semantic.py`** (`SemanticRetriever`):
   - Loads 384-dimensional dense embeddings (`catalogue.parquet` and `query_embeddings.parquet`).
   - Computes unit-normalised cosine similarity dot-products for intent-based candidates.

3. **`scorer_quality.py`** (`QualityScorer`):
   - Extracts product metadata (`rating`, `price`) from `catalogue.parquet`.
   - Computes static quality scores and provides candidate-level quality rankings.

4. **`hybrid_rrf_r1.py`** (Master Pipeline):
   - Unifies the candidate pools from BM25 and Semantic retrieval.
   - Applies 3-Way Reciprocal Rank Fusion (RRF with $k=60$):
     $$\text{RRF\_Score}(i) = \frac{1}{60 + \text{Rank}_{BM25}(i)} + \frac{1}{60 + \text{Rank}_{Sem}(i)} + w_{qual} \cdot \frac{1}{60 + \text{Rank}_{Qual}(i)}$$
   - Formats, validates (284,320 rows, 5 columns, 0-19 ranks), and submits the slate to the simulator API.
