# GSM Project Map 🗺️

> **Living document** — Single source of truth for the project's
> file/folder layout, key functions, and current conventions.
>
> Update this file whenever files are added/removed, functions are
> renamed, or defaults change. For coding standards, see
> [`.github/copilot-instructions.md`](.github/copilot-instructions.md).

---

## Pipeline Overview

```
GEO Expression Data + DisGeNET Gene-Disease Knowledge
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 GROUPING    SCORING     MODELING
 (Phase I)  (Phase II)  (Phase III)
    │           │           │
    ▼           ▼           ▼
 Disease-gene  Group      Final classifier +
 group         ranking    ranked features +
 projection    by ML      biological validation
```

**Entry point:** `src/workflows/GSM_workflow.py` → `gsm_workflow()`

---

## Root-Level Files

| File | Purpose |
|------|---------|
| `run_test.py` | Quick single-dataset pipeline run (dev/testing) |
| `run_all_datasets.py` | Batch-run the pipeline on all 7 GEO datasets |
| `dependencies.txt` | pip requirements |
| `pytest.ini` | Pytest configuration |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff, trailing whitespace) |
| `.gitattributes` | Git LFS tracking (`*.csv`) |
| `.lfsconfig` | LFS fetch-include for `data/` |
| `DATASET_EXCLUSIONS.md` | Why GDS3268 and GDS4206 were excluded |
| `CONTRIBUTING.md` | Contribution guidelines |
| `README.md` | Project overview and quick-start |
| `PROJECT_MAP.md` | **This file** |

---

## `src/` — Core Pipeline Code

### `src/data_processing/`
| File | Key functions |
|------|---------------|
| `data_loader.py` | `load_input_file()`, `load_group_file()`, `_detect_separator()` |
| `data_preprocess.py` | `preprocess_data()` |
| `normalization.py` | `normalize_expression()` |
| `handle_missing_values.py` | `impute_missing()` |
| `preliminary_filtering.py` | `preliminary_filter()` |
| `train_test_splitter.py` | `stratified_split()` |

### `src/feature_selection/`
| File | Key functions |
|------|---------------|
| `ttest_filter.py` | `ttest_filter_genes()` — Welch t-test + BH-FDR |
| `variance.py` | `variance_filter()` |
| `selectkbest.py` | `selectkbest_filter()` |
| `recursive_feature_elimination.py` | `rfe_filter()` |

### `src/grouping/` — Phase I
| File | Key functions |
|------|---------------|
| `run_grouping.py` | `run_grouping()` — orchestrator |
| `grouping_utils.py` | `build_gene_group_mapping()`, `filter_groups_by_size()` |
| `input_loader.py` | `load_grouping_input()` |

### `src/scoring/` — Phase II
| File | Key functions |
|------|---------------|
| `run_scoring.py` | `run_scoring()` — orchestrator |
| `feature_scorer.py` | `score_group()` |
| `score_data.py` | `ScoreData` dataclass |
| `metrics.py` | `compute_classification_metrics()` |
| `rank_aggreg.py` | `aggregate_group_ranks()` |

### `src/modeling/` — Phase III
| File | Key functions |
|------|---------------|
| `run_modeling.py` | `run_modeling()` — orchestrator |
| `evaluator.py` | `evaluate_model()` |
| `modeling_utils.py` | `select_top_groups()`, `pool_group_features()` |

### `src/machine_learning/`
| File | Key functions |
|------|---------------|
| `classification.py` | `get_classifier()` — factory |
| `random_forest.py` | `create_random_forest()` |
| `xgboost.py` | `create_xgboost_classifier()` |
| `adaboost.py` | `create_adaboost()` |
| `decision_tree.py` | `create_decision_tree()` |
| `logitboost.py` | `create_logitboost()` |

### `src/utils/`
| File | Key functions |
|------|---------------|
| `logger.py` | `setup_logger()` |
| `save_results.py` | `save_iteration_results()`, `save_summary_json()` |
| `save_ranked_features.py` | `save_ranked_features()` |
| `save_ranked_groups.py` | `save_ranked_groups()` |
| `rank_aggregation.py` | `robust_rank_aggregation()` — RRA (Stuart et al.) |
| `biological_validation.py` | `run_biological_validation()` — Enrichr + STRING-db + DisGeNET |
| `generate_figures.py` | `generate_all_figures()` |
| `visualization.py` | `plot_roc_curve()`, `plot_feature_importance()` |
| `performance_profiler.py` | `profile_pipeline()` |

### `src/workflows/`
| File | Key functions |
|------|---------------|
| `GSM_workflow.py` | `gsm_workflow()` — **main entry point** |
| `GSM_workflow_config.py` | `GSMConfig` dataclass |
| `classification_workflow.py` | `classification_workflow()` — plain classification baseline |
| `classification_workflow_config.py` | `ClassificationConfig` dataclass |
| `group_lasso_workflow.py` | `group_lasso_workflow()` — Group Lasso with overlap-aware feature duplication, multi-iteration evaluation, and integrated biological validation (Enrichr + STRING-db) |
| `group_lasso_workflow_config.py` | `GroupLassoConfig` dataclass — includes `run_biological_validation`, `biological_validation_top_genes`, `disgenet_api_key`, `max_groups_per_gene`, `min_genes_per_group` |
| `stability_selection_gl.py` | `run_stability_selection()` — **Novel**: combines Meinshausen & Bühlmann (2010) stability selection with Latent Group Lasso; B subsamples, per-gene Π̂ probabilities, provable FDR upper bound via Theorem 1 |
| `gl_rf_hybrid.py` | `gl_rf_hybrid_workflow()` — **Novel**: Two-Stage GL→RF Hybrid Pipeline; Stage 1 = Group Lasso feature pre-selection, Stage 2 = Random Forest classification on GL-selected genes |

### `src/ui/`
| File | Purpose |
|------|---------|
| `app.py` | Streamlit web GUI |

---

## `scripts/` — Analysis & Manuscript Tooling

| File | Purpose | Output location |
|------|---------|-----------------|
| `analyze_manuscript_results.py` | Aggregate metrics → JSON + figures | `reports_ARCHIVE/manuscript_data.json`, `reports_ARCHIVE/manuscript_figures/` |
| `build_manuscript_docx.py` | Generate Word manuscript | `reports_ARCHIVE/GSM_Manuscript_v*.docx` |
| `generate_flowchart.py` | Figure 1: pipeline flowchart | `reports_ARCHIVE/manuscript_figures/` |
| `generate_baseline_comparison.py` | Baseline comparison figure | `reports_ARCHIVE/manuscript_figures/` |
| `compare_gl_vs_gsm.py` | GSM vs Group Lasso side-by-side comparison (metrics, gene overlap, figures) | `output/comparison_gl_vs_gsm/` |
| `gl_ablation_study.py` | Compare 3 overlap strategies: duplication (=Latent GL), naive single-group, no-group L1 | `output/gl_ablation_<dataset>_<ts>/` |
| `run_gl_all_datasets.py` | Run GL workflow on all 7 cancer datasets with cross-dataset summary | `output/gl_all_datasets_<ts>/` |
| `gl_hyperparameter_search.py` | Grid search over λ₁, λ₂ and filter thresholds with heatmap output | `output/gl_hyperparam_<dataset>_<ts>/` |
| `MANUSCRIPT_BUILD_STEPS.txt` | Build order documentation for GSM manuscript | — |
| `run_baselines.py` | Non-GSM baseline classifiers | `output/baselines/baseline_results.json` |
| `run_sensitivity_analysis.py` | One-at-a-time sensitivity analysis | `output/sensitivity_runs/sensitivity_results.json` |
| `compute_sensitivity_impact.py` | Δ-F1 impact rankings | `output/sensitivity_runs/sensitivity_impact.json` |
| `benchmark_scoring_models.py` | Benchmark 11 ML models for scoring | `output/benchmark/benchmark_results.json` |
| `compare_classifiers_bio_validation.py` | XGBoost vs RF bio coherence | `output/classifier_comparison/classifier_comparison_results.json` |
| `extend_classifier_comparison.py` | Extended 4-classifier comparison | `output/classifier_comparison/classifier_comparison_results.json` |
| `seed_stability_experiment.py` | Seed stability analysis | `output/seed_stability/seed_stability_results.json` |
| `rerun_bio_validation.py` | Re-run failed bio validations | In-place in `output/<run>/biological_validation/` |
| `rerun_seed_bio_validation.py` | Re-run seed bio validations | `output/seed_stability/seed_stability_results.json` |
| `verify_gene_lists.py` | Sanity-check gene lists | stdout |

---

## `data/`

| Folder | Contents |
|--------|----------|
| `data/main_data/` | GEO expression matrices (`.csv`, tracked by Git LFS) |
| `data/grouping_data/` | DisGeNET gene-disease mappings |
| `data/test/` | Small test fixtures for unit tests |
| `data/data_ARCHIVE/` | Archived/deprecated datasets |

---

## `tests/`

| File | Coverage |
|------|----------|
| `test_core_functions.py` | 29 tests: data loading, t-test, grouping, scoring, modeling, rank aggregation, bio validation, profiling |

---

## `output/` — Pipeline Outputs (gitignored)

Each pipeline run creates a timestamped folder. Organized subdirectories:

| Subfolder | Contents |
|-----------|----------|
| `main_runs/` | Production pipeline runs (7 datasets × RF) |
| `glasso_<timestamp>/` | Group Lasso runs (includes `group_lasso_results.json`, `gene_selection_frequency.xlsx`, `biological_validation/`) |
| `stability_test/` or `stability_gl_<ts>/` | Stability selection runs (`stability_results.json`, `stability_gene_probabilities.xlsx`, probability figures) |
| `gl_rf_hybrid_<dataset>_<ts>/` | GL→RF hybrid runs (`gl_rf_hybrid_results.json`, `hybrid_gene_frequency.xlsx`, comparison figures) |
| `comparison_gl_vs_gsm/` | GSM vs GL comparison (metrics, gene overlap, figures) |
| `gl_ablation_<dataset>_<ts>/` | Ablation study: duplication vs naive vs no-group L1 |
| `gl_all_datasets_<ts>/` | Multi-dataset GL evaluation with cross-dataset summary |
| `gl_hyperparam_<dataset>_<ts>/` | Hyperparameter grid search results and heatmaps |
| `classifier_comparison/` | XGBoost vs RF and extended classifier runs |
| `seed_stability/` | Seed stability experiment results |
| `sensitivity_runs/` | Sensitivity analysis runs + results JSON |
| `baselines/` | Baseline comparison results |
| `benchmark/` | Scoring model benchmark results |
| `archives/` | Old compressed runs |

---

## `reports_ARCHIVE/` — Manuscript Artefacts (gitignored)

`manuscript_data.json`, `manuscript_figures/`, generated `.docx` files,
supplementary PDFs, and related publications.

| Subfolder | Contents |
|-----------|----------|
| `group_lasso_manuscript/` | GL manuscript builder (`build_gl_manuscript.py`) and generated `GL_Manuscript_v*.docx` files. Accepts `--comparison-dir` for GSM vs GL data and reads bio validation results automatically. |

---

## `DOCS/` — Documentation

| File | Topic |
|------|-------|
| `DEVELOPMENT.md` | Project structure & coding standards |
| `RUNNING.md` | How to run the pipeline |
| `INSTALL_WSL.md` | WSL/Linux setup |

| `GITHUB_WORKFLOW.md` | Branching & PR guidelines |
| `COPILOT.md` | GitHub Copilot usage |
| `TROUBLESHOOTING.md` | Common fixes |

---

## Key Conventions

| Parameter | Value | Notes |
|-----------|-------|-------|
| Default classifier | Random Forest | seed = 44, deterministic derivation per iteration |
| Iterations | 100 per run | Each iteration: new train/test split |
| FDR threshold | α = 0.05 | Welch t-test + Benjamini–Hochberg correction |
| CV folds (scoring) | 3-fold stratified | Phase II group evaluation |
| CV folds (validation) | 5-fold stratified | Phase III final model |
| Feature ranking | Robust Rank Aggregation | Stuart et al. method |
| Biological validation | Enrichr + STRING-db + DisGeNET | Optional, external APIs |
| Excluded datasets | GDS3268 (breast), GDS4206 (HCC) | See `DATASET_EXCLUSIONS.md` |
| Supported classifiers | RF, XGBoost, DecisionTree, SVM, KNN, MLP | Via `get_classifier()` factory |
| Python version | 3.10+ | 3.11 or 3.12 recommended |

---

*Last updated: 2026-03-06*
