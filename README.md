# Single recorded maltreatment type and type-defined research samples

Code supporting an unpublished manuscript.

The study compares one recorded maltreatment type with four criterion-based domain indicators in 3,236 AI–Hub child counseling session records. Record-level agreement and type-specific sample composition are distinct quantities. The code does not validate a clinical diagnosis or estimate maltreatment prevalence in the general population.

## Scope

This is the manuscript's analysis and figure-generation repository. It contains no counseling records, participant identifiers, counseling narratives, per-record scores, individual predictions, previous modeling projects, private review conversations, manuscript drafts or PDFs. The small CSV/JSON inputs under `analysis/*_outputs/` are aggregate statistics supporting the paper; rows identify categories or joint cells, never people. Existing model estimates are archived aggregates, not a claim that every model can be refitted without source access.

The release tag `paper-2026-09-28` marks the released code version. No DOI has been assigned. The manuscript has not been represented here as accepted or published.

## Environment and aggregate reproduction

The analysis used Python 3.11.5. Install `requirements.txt` in a suitable Python environment, then run from this directory:

```bash
python -m pip install -r requirements.txt
python tools/reproduce_aggregates.py
python -m unittest discover -s analysis -p 'test_*.py'
python tools/build_rq2_figures.py
python analysis/make_benchmark_domain_figure.py
python analysis/make_profile_figure.py
python analysis/make_score_figures.py --input analysis/selection_outputs/score_distribution.csv --out-dir generated/figures/ko --language ko
python analysis/make_score_figures.py --input analysis/selection_outputs/score_distribution.csv --out-dir generated/figures/en --language en
python tools/verify_release.py
```

The paired confidence intervals use 9,999 multinomial resamples of the complete 16-profile by 5-recorded-type frequency table, with seed 20260914. For these statistics, joint-cell resampling is equivalent to resampling whole records and preserves overlap between the two sample definitions. The four RQ2 panels show sample sizes, Jaccard overlap, co-occurring-domain percentages, and their differences with 95% intervals. Percentages and percentage-point differences are separate units.

The reproduction command writes fresh outputs under `generated/` and compares them with the frozen inputs. Figure commands write Korean and English plots there. The bundled font allows Korean plots on systems without Apple fonts. Generated files are excluded from Git.

## Reproduction requiring authorized source access

Acquire the source data under AI–Hub's access terms. Do not put source files in this repository. Use an external local directory and supply its path explicitly:

```bash
python analysis/sensitivity_analysis.py --train-dir /authorized/training --validation-zip /authorized/VL_out.zip --out-dir generated/full_analysis
python analysis/selection_audit.py --help
python analysis/measurement_audit.py --training-zip /authorized/TL_out.zip --validation-zip /authorized/VL_out.zip --out-dir generated/measurement
python analysis/anchor_counselor_audit.py --source-dir /authorized/extracted_json --joint-table analysis/sensitivity_outputs/combination_record_matrix.csv --out-dir generated/anchor_checks
python tools/measurement_checks.py --source-dir /authorized/extracted_json
python tools/tie_resolution_checks.py --source-dir /authorized/extracted_json
```

The last two commands reproduce the cutoff/allocation checks and tie handling. Run them in that order because the second expands the tie-handling table written by the first. Severity-based allocation, ordered/conditional choice models, child-group and stage checks need information unavailable in the public joint cells. These scripts process restricted inputs in memory and export aggregates only. Their existence does not imply that a full raw-data rerun was performed during repository preparation.

## Manuscript correspondence

| Paper component | Code and aggregate input |
|---|---|
| RQ1 agreement and criterion-meeting profiles | `reproduce_representation_audit.py`, `sensitivity_analysis.py`, `sensitivity_outputs/combination_record_matrix.csv` |
| RQ2 sample inclusion/exclusion and co-occurring concerns | `cooccurrence_benchmarks.py`, `sample_composition_uncertainty.py`, `sample_components.py`, `composition_ci_outputs/joint_counts.csv` |
| RQ3 supplementary allocation rules and score position | `selection_audit.py`, `anchor_counselor_audit.py`, `tools/measurement_checks.py`, `tools/tie_resolution_checks.py` |
| Figure 1, Korean and English | `tools/build_rq2_figures.py` |
| Figure 2 | `make_benchmark_domain_figure.py` |
| Figure 3, UpSet with recorded-type stacked bars | `make_profile_figure.py` |
| Figure 4, score distributions | `make_score_figures.py` |
| Supplemental co-occurrence models | `sensitivity_analysis.py`, `sensitivity_outputs/*loglinear*`, `sensitivity_outputs/*pairwise*` |
| Supplementary tie comparison and score-anchor mapping | `tools/tie_resolution_checks.py`, `tie_resolution/` |

Main figure numbering follows the current manuscript. The two LaTeX tables in `checks/manuscript_tables/` are fixed numerical checks for the RQ2 renderer, not full manuscript sources. `MANIFEST.json` lists SHA-256 hashes of all release files. Statistical calculations are preserved from the verified local analysis; repository paths, output directories and font handling have been made portable. Old table/prose builders are omitted.

AI assistance: OpenAI Codex and Claude assisted with code development and manuscript editing. The author remains responsible for the data, analyses and manuscript. External research funding was not received. Ethics determination is separate from technical reproducibility.
