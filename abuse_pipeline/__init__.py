"""Refactored v28 pipeline package.

Subpackages:
  core/          - common config, label classification, text processing, plots
  data/          - document-level aggregation, embeddings, counting
  stats/         - statistical analysis, correspondence analysis
  classifiers/   - ML classifiers (TF-IDF, BERT)
  analysis/      - sensitivity analysis, label comparison
  investigation/ - exploratory analysis (no-GT, borderline)
  revision/      - revision-specific extensions

Entry point: run_v28_refactor.py at project root.
"""

# ── Backward-compatible re-exports ────────────────────────────────
# Allow old-style imports like `from abuse_pipeline import common`
# and `from abuse_pipeline.common import *` to continue working.
from abuse_pipeline.core import common
from abuse_pipeline.core import labels
from abuse_pipeline.core import text
from abuse_pipeline.core import plots

from abuse_pipeline.data import doc_level
from abuse_pipeline.data import embedding
from abuse_pipeline.data import counting

from abuse_pipeline.stats import stats
from abuse_pipeline.stats import ca
from abuse_pipeline.stats import bridge_threshold_justification
from abuse_pipeline.stats import weighted_ca_extension
from abuse_pipeline.stats import contextual_embedding_ca

from abuse_pipeline.classifiers import classifier_utils
from abuse_pipeline.classifiers import tfidf_vs_bert_comparision
from abuse_pipeline.classifiers import neg_gt_multilabel_analysis
from abuse_pipeline.classifiers import bert_hyperparameter
from abuse_pipeline.classifiers import bert_abuse_coloring

from abuse_pipeline.analysis import compare_abuse_labels
from abuse_pipeline.analysis import label_comparsion_analysis
from abuse_pipeline.analysis import main_sub_abuse_analysis
from abuse_pipeline.analysis import integrated_label_bridge_analysis
from abuse_pipeline.analysis import main_threshold_sensitivity
from abuse_pipeline.analysis import sub_threshold_sensitivity

from abuse_pipeline.investigation import no_gt
from abuse_pipeline.investigation import borderline_case_explorer

from abuse_pipeline.revision import revision_extensions
from abuse_pipeline.revision import revision_v2

# ── sys.modules shim ──────────────────────────────────────────────
# Allow `from abuse_pipeline.common import X` (module-path style)
# to work even though common.py now lives at core/common.py.
import sys as _sys

_module_map = {
    "abuse_pipeline.common": common,
    "abuse_pipeline.labels": labels,
    "abuse_pipeline.text": text,
    "abuse_pipeline.plots": plots,
    "abuse_pipeline.doc_level": doc_level,
    "abuse_pipeline.embedding": embedding,
    "abuse_pipeline.counting": counting,
    "abuse_pipeline.stats": stats,
    "abuse_pipeline.ca": ca,
    "abuse_pipeline.bridge_threshold_justification": bridge_threshold_justification,
    "abuse_pipeline.weighted_ca_extension": weighted_ca_extension,
    "abuse_pipeline.contextual_embedding_ca": contextual_embedding_ca,
    "abuse_pipeline.classifier_utils": classifier_utils,
    "abuse_pipeline.tfidf_vs_bert_comparision": tfidf_vs_bert_comparision,
    "abuse_pipeline.neg_gt_multilabel_analysis": neg_gt_multilabel_analysis,
    "abuse_pipeline.bert_hyperparameter": bert_hyperparameter,
    "abuse_pipeline.bert_abuse_coloring": bert_abuse_coloring,
    "abuse_pipeline.compare_abuse_labels": compare_abuse_labels,
    "abuse_pipeline.label_comparsion_analysis": label_comparsion_analysis,
    "abuse_pipeline.main_sub_abuse_analysis": main_sub_abuse_analysis,
    "abuse_pipeline.integrated_label_bridge_analysis": integrated_label_bridge_analysis,
    "abuse_pipeline.main_threshold_sensitivity": main_threshold_sensitivity,
    "abuse_pipeline.sub_threshold_sensitivity": sub_threshold_sensitivity,
    "abuse_pipeline.no_gt": no_gt,
    "abuse_pipeline.borderline_case_explorer": borderline_case_explorer,
    "abuse_pipeline.revision_extensions": revision_extensions,
    "abuse_pipeline.revision_v2": revision_v2,
}
for _key, _mod in _module_map.items():
    _sys.modules.setdefault(_key, _mod)
del _sys, _module_map, _key, _mod
