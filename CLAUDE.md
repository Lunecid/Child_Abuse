# CLAUDE.md

## Project Overview

This is a **child abuse detection and analysis research pipeline** (v28 Refactor) for an academic paper. It processes structured JSON records from Korean child counseling sessions to:

1. **Classify children** into emotional valence groups (부정/평범/긍정 = Negative/Neutral/Positive)
2. **Identify abuse types** (성학대/신체학대/정서학대/방임 = Sexual/Physical/Emotional/Neglect)
3. **Perform NLP analysis** on children's speech transcripts using Korean morphological analysis
4. **Compute statistical measures** (log-odds, chi-square, correspondence analysis) to identify discriminative vocabulary
5. **Identify "bridge words"** — tokens spanning multiple abuse types, indicating co-occurrence patterns
6. **Run ML classifiers** (TF-IDF + LR/RF/SVM, KLUE-BERT) for abuse type prediction
7. **Generate publication-ready outputs** (CSV tables, word clouds, radar charts, CA biplots)

## Repository Structure

```
Child_Abuse_Lab/
├── run_v28_refactor.py                    # Main entry point: runs full pipeline (ALL + NEG_ONLY)
├── run_neg_gt_multilabel.py               # Standalone: multi-label vs single-label classifier comparison
├── run_softlabel_vs_singlelabel.py        # Standalone: soft-label vs single-label analysis
├── run_gt_alg_gap_diagnosis.py            # Standalone: GT-Algorithm gap diagnosis
├── run_abuse_neg_rebuttal_metrics.py      # Standalone: negative-group rebuttal metrics
├── extract_threshold_statistics.py        # Standalone: threshold justification statistics for appendix
├── data/                                  # Input JSON files (not committed; place *.json here)
│
└── abuse_pipeline/                        # Main Python package
    ├── __init__.py                        # Backward-compatible re-exports + sys.modules shim
    ├── pipeline.py                        # Orchestrator: 10-stage pipeline (run_pipeline())
    │
    ├── core/                              # Core utilities (shared across all modules)
    │   ├── common.py                      # Global config, constants, thresholds, TFIDF_PARAMS
    │   ├── labels.py                      # classify_child_group(), classify_abuse_main_sub()
    │   ├── text.py                        # extract_child_speech(), tokenize_korean(), bridge utterance extraction
    │   └── plots.py                       # Visualization: radar charts, treemaps, TF-IDF logistic regression
    │
    ├── data/                              # Data processing modules
    │   ├── counting.py                    # GT-based abuse type counting (raw labels only)
    │   ├── doc_level.py                   # Document-level frequency tables, permutation tests, bootstrap
    │   └── embedding.py                   # Word2Vec / FastText training and projection
    │
    ├── stats/                             # Statistical analysis
    │   ├── stats.py                       # HHI, cosine similarity, chi-square, log-odds, BH-FDR, bridge detection
    │   ├── ca.py                          # Correspondence Analysis (CA) with bridge word overlays
    │   ├── bridge_threshold_justification.py  # 3-strategy bridge threshold validation
    │   ├── contextual_embedding_ca.py     # BERT embedding CA validation (Procrustes + Mantel)
    │   └── weighted_ca_extension.py       # Weighted CA with sub-abuse type contributions
    │
    ├── classifiers/                       # Machine learning classifiers
    │   ├── classifier_utils.py            # Shared TF-IDF + sklearn / BERT fine-tuning utilities
    │   ├── tfidf_vs_bert_comparision.py   # TF-IDF vs BERT classifier comparison
    │   ├── neg_gt_multilabel_analysis.py  # Multi-label (main+sub) vs single-label comparison
    │   ├── softlabel_vs_singlelabel_analysis.py  # Soft-label vs single-label analysis
    │   ├── bert_hyperparameter.py         # BERT hyperparameter grid search (9 combos)
    │   └── bert_abuse_coloring.py         # BERT word-level abuse type coloring
    │
    ├── analysis/                          # Extended analysis modules
    │   ├── compare_abuse_labels.py        # GT vs algorithm label comparison
    │   ├── label_comparsion_analysis.py   # Label comparison statistics
    │   ├── main_sub_abuse_analysis.py     # Main + sub abuse type analysis
    │   ├── integrated_label_bridge_analysis.py  # 7-stage integrated analysis (standalone capable)
    │   ├── main_threshold_sensitivity.py  # Main abuse threshold sensitivity
    │   ├── sub_threshold_sensitivity.py   # Sub abuse threshold sensitivity
    │   ├── gt_alg_gap_diagnosis.py        # GT-Algorithm gap diagnosis
    │   └── abuse_neg_rebuttal_metrics.py  # Negative-group rebuttal metrics
    │
    ├── investigation/                     # Exploratory / diagnostic modules
    │   ├── borderline_case_explorer.py    # Borderline cases (0 < max(A_k) <= 6)
    │   └── no_gt.py                       # Cases with no ground-truth labels
    │
    └── revision/                          # Paper revision extensions
        ├── revision_extensions.py         # Threshold sensitivity, FDR re-report, multi-classifier comparison
        └── revision_v2.py                 # Appendix: soft labels, preprocessing robustness, GT bias check
```

## How to Run

### Full Pipeline
```bash
python run_v28_refactor.py [--data_dir /path/to/json/files]
```
- Reads `*.json` from `data/` (or specified directory)
- Runs pipeline twice: ALL children, then NEG_ONLY (negative valence only)
- Outputs to `ver28_all/` and `ver28_negOnly/`

### Standalone Scripts
```bash
python run_neg_gt_multilabel.py --data_dir /path/to/data [--skip_bert]
python run_softlabel_vs_singlelabel.py --data_dir /path/to/data
python run_gt_alg_gap_diagnosis.py --data_dir /path/to/data
python run_abuse_neg_rebuttal_metrics.py --data_dir /path/to/data
python extract_threshold_statistics.py
```

## Data Format

Input: one JSON file per child with this structure:
```json
{
  "info": {
    "ID": "...", "합계점수": 30, "위기단계": "상담필요",
    "성별": "...", "나이": "...", "학년": "...",
    "임상진단": "...", "임상가 종합소견": "...", "학대의심": "..."
  },
  "list": [
    {
      "문항": "학대여부", "문항합계": 12,
      "list": [
        {
          "항목": "정서학대", "점수": 5,
          "audio": [
            {"type": "A", "text": "child's spoken response"},
            {"type": "Q", "text": "counselor's question"}
          ]
        }
      ]
    }
  ]
}
```

Key fields:
- `info.위기단계` (crisis stage): 응급, 위기아동, 학대의심, 상담필요, 관찰필요, 정상군
- `audio[].type`: "A" = child answer, "Q" = counselor question

## Key Domain Concepts

### Valence Groups (정서군)
- **부정 (Negative)**: At-risk or confirmed abuse
- **평범 (Neutral)**: Borderline or mixed indicators
- **긍정 (Positive)**: No significant risk

Classification: rule-based algorithm in `labels.py:classify_child_group()` using total score, crisis stage, risk score, and protective factors.

### Abuse Types (학대유형) — severity-ordered
1. **성학대 (Sexual)** — rank 0 (most severe, wins ties)
2. **신체학대 (Physical)** — rank 1
3. **정서학대 (Emotional)** — rank 2
4. **방임 (Neglect)** — rank 3

Each child gets a **main** type (score >6) and optional **sub** types (score >=4). Ties broken by severity rank.

### Bridge Words (교량 단어)
Words appearing across multiple abuse types with similar probability:
- `p1` (top P(abuse|word)) >= 0.40, `p2` (2nd) >= 0.25, `gap` (p1-p2) <= 0.20

## ML Model Hyperparameters

### TF-IDF (Global: `core/common.py` → `TFIDF_PARAMS`)

All classifier modules reference this single dict for consistency:

```python
TFIDF_PARAMS = dict(
    tokenizer    = str.split,      # Pre-tokenized text (Okt output), whitespace split
    preprocessor = None,           # No sklearn preprocessing (already Okt-processed)
    token_pattern= None,           # Must be None when custom tokenizer is used
    ngram_range  = (1, 2),         # Unigram + bigram features
    min_df       = 2,              # Minimum 2 documents (auto-relaxed to 1 if no terms)
    max_features = 20000,          # Top 20k features by TF-IDF score
)
```

Used in: `classifier_utils.py`, `tfidf_vs_bert_comparision.py`, `plots.py`, `revision_v2.py`, `softlabel_vs_singlelabel_analysis.py`

### TF-IDF Classifiers (Single-label)

| Classifier | Class | Key Params |
|---|---|---|
| LR | `LogisticRegression` | `solver="lbfgs"`, `max_iter=300` |
| RF | `RandomForestClassifier` | `n_estimators=200`, `max_depth=None` |
| SVM | `LinearSVC` | `max_iter=1000`, `dual="auto"` |

### TF-IDF Classifiers (Multi-label / Binary Relevance)

| Classifier | Class | Key Params |
|---|---|---|
| LR | `LogisticRegression` | `solver="liblinear"`, `max_iter=1000`, `class_weight="balanced"` |
| RF | `RandomForestClassifier` | `n_estimators=200`, `class_weight="balanced"` |
| SVM | `LinearSVC` | `max_iter=1000`, `class_weight="balanced"` |

### TF-IDF + Multinomial Logistic Regression (Stage 6, `plots.py`)

```python
LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=300)
```

### KLUE-BERT Fine-tuning

| Param | Default | Grid Search Range |
|---|---|---|
| `model_name` | `klue/bert-base` | — |
| `max_length` | 256 | — |
| `batch_size` | 16 | — |
| `epochs` | 10 | [3, 5, 10] |
| `learning_rate` | 2e-5 | [1e-5, 2e-5, 5e-5] |
| `weight_decay` | 0.01 | — |
| `warmup_ratio` | 0.1 (of total steps) | — |
| `grad_clip` | 1.0 | — |

### Evaluation Protocol

- Stratified K-Fold CV: `n_splits=5`, `shuffle=True`, `random_state=42`
- Metrics: Accuracy, Macro F1, Weighted F1, Cohen's κ, per-label P/R/F1
- Confusion matrix + ambiguity zone analysis (cross-classifier disagreement)

## Architecture Patterns

### Global Configuration (`core/common.py`)
- All thresholds and constants centralized
- Output dirs set dynamically via `configure_output_dirs(subset_name, base_dir, version_tag)`
- Optional dependency flags (`HAS_SCIPY`, `HAS_SKLEARN`, `HAS_GENSIM`, `HAS_TRANSFORMERS`, `HAS_OKT`) for graceful degradation
- `BRIDGE_P_CONFIGS` defines baseline/loose/strict parameter sets
- `TFIDF_PARAMS` ensures consistency across classifier modules

### Korean NLP (`core/text.py`)
- Morphological analysis via `konlpy.Okt` with stemming
- POS filter: Nouns, Verbs, Adjectives only
- Negation bigrams: "안/못 + verb" → "안_먹다"; "않다" → prefix merge
- Domain stopwords in `STOPWORDS_BASE`; single-char tokens dropped
- Fallback to whitespace tokenization if Okt unavailable

### Backward Compatibility (`__init__.py`)
- Module re-exports so `from abuse_pipeline import common` works
- `sys.modules` shim for old paths like `from abuse_pipeline.common import X`

## Important Thresholds

| Constant | Value | Purpose |
|---|---|---|
| `high_total_thresh` | 45 | Total score → forced negative |
| `low_total_thresh` | 10 | Total score → positive (if no other signal) |
| `risk_strong` / `risk_weak` | 25 / 10 | Risk score thresholds |
| Abuse main threshold | >6 | Qualifies as main abuse type |
| `sub_threshold` | 4 | Sub-type assignment |
| `MIN_TOTAL_COUNT_VALENCE` | 10 | Min token freq for valence analysis |
| `MIN_TOTAL_COUNT_ABUSE` | 8 | Min token freq for abuse analysis |
| `MIN_DOC_COUNT` | 5 | Min document freq for doc-level analysis |
| `BRIDGE_MIN_P1/P2/MAX_GAP` | 0.40/0.25/0.20 | Bridge word selection |
| `DEFAULT_ZMIN` | 1.96 | Z-score threshold (alpha=0.05) |

## Dependencies

### Required
- Python 3.9+, `numpy`, `pandas`, `matplotlib`

### Optional (graceful degradation when missing)
- `scipy` — Statistical tests (ANOVA, chi-square, Spearman)
- `scikit-learn` — TF-IDF, classifiers (LR, RF, SVM), cross-validation
- `konlpy` — Korean morphological analysis (Okt)
- `gensim` — Word2Vec / FastText
- `prince` — Correspondence Analysis
- `transformers` + `torch` — KLUE-BERT fine-tuning
- `wordcloud`, `squarify`, `adjustText`, `statsmodels`, `deep_translator`

## Development Conventions

### Language
- **Code**: Python with English names; **comments/docstrings**: Korean + English mix
- **Console output**: Korean; **CSV encoding**: `utf-8-sig` (Excel-safe Korean)

### Code Style
- `from __future__ import annotations` used consistently
- Type hints with `|` union syntax (Python 3.10+)
- Modules use `from abuse_pipeline.core.common import *` for shared constants
- Matplotlib backend: `"Agg"` (non-interactive, file-only)

### Reproducibility
- Tie-breaking via `SEVERITY_RANK` (deterministic child-safety-first hierarchy)
- Random seed: 42 throughout
- Stable sort (`kind="mergesort"`) for chi-square rankings

### Output Organization
```
ver28_all/ or ver28_negOnly/
├── meta/               # Preprocessing examples, stopwords, distributions
├── valence/stats/      # Log-odds, chi-square, HHI, TF-IDF logit
├── valence/figures/    # Radar charts, word clouds
├── abuse/stats/        # Abuse statistics, bridge words, freq-matched baselines
├── abuse/figures/      # Word clouds, CA biplots
├── embeddings/         # Word2Vec/FastText models + projections
├── pbridge_ablation/   # Bridge word ablation
├── bert_ca_validation/ # BERT CA + Procrustes
├── revision/           # Paper revision outputs
└── neg_gt_multilabel/  # Multi-label classifier comparison (NEG_ONLY only)
```

## Common Tasks

### Adding a New Analysis Module
1. Create under appropriate subpackage (`analysis/`, `stats/`, etc.)
2. Import from `abuse_pipeline.core.common`, `core.labels`, `core.text`
3. Add import to `abuse_pipeline/__init__.py` (both direct + `_module_map`)
4. Call from `pipeline.py` or make standalone-capable

### Modifying Thresholds
All in `core/common.py`. Sensitivity analyses exist:
- `analysis/main_threshold_sensitivity.py` and `sub_threshold_sensitivity.py`
- `stats/bridge_threshold_justification.py`
