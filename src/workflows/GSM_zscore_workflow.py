"""
GSM_zscore_workflow.py
----------------------

Bu dosya, orijinal GSM_workflow.py ile aynı adımları izler; tek fark,
özellik seçim (feature_selection) adımlarından hemen sonra
gsm_with_zscore.py çalıştırılır, ardından GSM akışının geri kalanı
aynı sırayla devam eder.

Notlar:
- GSM_workflow.py veya feature_selection modüllerine dokunulmaz.
- Z-score adımı, preliminary t-test filtrelemesinin hemen sonrasında,
  gruplaya (grouping) geçmeden önce eklenir.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import time
import logging
import random

import pandas as pd 
import numpy as np

import src.workflows.gsm_with_zscore as gsm_zscore
import src.workflows.GSM_zscore_workflow_config as config

# Aynı yardımcıları ve fonksiyonları kullanmak için orijinal workflow importları
from src.workflows.GSM_workflow import (
    format_duration,
    _write_progress,
    save_runtime_config,
    build_runtime_log_items,
    log_runtime_config,
    generate_iteration_seed,
    set_random_seed,
    save_results,
    visualize_f1_scores,
    generate_all_figures,
    aggregate_group_ranks_rra,
    aggregate_feature_ranks_rra,
    load_ranked_groups_from_excel,
    load_ranked_features_from_excel,
    save_aggregated_group_ranking,
    save_aggregated_feature_ranking,
    save_best_averaged_rankings,
    compute_best_averaged_groups,
    compute_best_averaged_features,
    aggregate_model_feature_importance_rra,
    ModelArtifact,
    save_model_bundle,
    expand_groups_until_feature_increase,
)

from src.utils.logger import setup_logger
from src.data_processing.data_loader import load_input_file, load_group_file
from src.data_processing.data_preprocess import preprocess_data, preprocess_grouping_data
from src.data_processing.train_test_splitter import split_data
from src.data_processing.preliminary_filtering import preliminary_ttest_filter
from src.scoring.run_scoring import run_scoring, score_all_features
from src.grouping.run_grouping import run_grouping
from src.modeling.run_modeling import ModelingResult, run_modeling, select_features_from_top_groups
from src.scoring.feature_scorer import FeatureScore
from src.scoring.metrics import MetricsData
from src.utils.save_ranked_features import save_ranked_features, FeatureRankingOutput
from src.utils.visualization import visualize_f1_scores
from src.data_processing.normalization import fit_scaler
from src.workflows.postprocessing_helpers import ensure_results_json, run_postprocessing_aggregations
from src.utils.rank_aggregation import (
    RankedFeatureItem,
    RankedFeatureList,
    RankedGroupItem,
    RankedGroupList,
    aggregate_group_ranks_rra,
    aggregate_feature_ranks_rra,
    save_aggregated_group_ranking,
    save_aggregated_feature_ranking,
    compute_best_averaged_groups,
    compute_best_averaged_features,
    save_best_averaged_rankings,
    aggregate_model_feature_importance_rra,
)


# -----------------------------------------------------------------------------
# Z-Score entegrasyonlu ana döngü
# -----------------------------------------------------------------------------


def gsm_main_loop_zscore(
    data: pd.DataFrame,
    grouping_data: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    iteration: int,
    logger,
    *,
    gene_column: str = config.GENE_COLUMN_NAME,
    group_column: str = config.GROUP_COLUMN_NAME,
    label_column: str = config.LABEL_COLUMN_NAME,
    sample_ratio: float = config.TRAIN_TEST_SPLIT_RATIO,
    ttest_threshold: float = config.TTEST_THRESHOLD,
    q_value_threshold: float = config.Q_VALUE_THRESHOLD,
    initial_feature_filter_size: int = config.INITIAL_FEATURE_FILTER_SIZE,
    best_groups_to_keep: int = config.BEST_GROUPS_TO_KEEP,
    cross_validation_folds: int = config.CROSS_VALIDATION_FOLDS,
    scoring_model: str = config.SCORING_MODEL,
    iteration_seed: int = config.RANDOM_SEED,
    save_group_derived_features: bool = False,
) -> Tuple[List[ModelingResult], List[MetricsData]]:
    """
    GSM_workflow.gsm_main_loop'un aynısı; tek fark, t-test temelli
    feature selection sonrası z-score pipeline çağrılır.
    """
    logger.info("##### Starting GSM Main Loop (z-score) #####")

    # Data Splitting
    test_size = round(1.0 - sample_ratio, 4)
    logger.info(f"Split {sample_ratio:.0%}/{test_size:.0%} (seed={iteration_seed})")
    train_test_split_data = split_data(
        data, label_column, test_size=test_size, stratify=True, random_state=iteration_seed
    )

    # Feature Filtering (on training data only — no data leakage)
    filtered_train = preliminary_ttest_filter(
        train_test_split_data.X_train,
        train_test_split_data.y_train,
        threshold=ttest_threshold,
        initial_feature_filter_size=initial_feature_filter_size,
        logger=logger,
    )

    # Apply z-score filtering on the current training split to avoid leakage and duplicate I/O.
    logger.info("Applying in-memory z-score group filtering on training split...")
    if filtered_train.selected_feature_names is not None:
        filtered_feature_names = list(filtered_train.selected_feature_names)
    else:
        filtered_feature_names = list(
            train_test_split_data.X_train.columns[filtered_train.selected_features]
        )

    try:
        zscore_filtered_feature_names, significant_group_df = gsm_zscore.run_zscore_filter(
            filtered_train,
            list(train_test_split_data.X_train.columns),
            grouping_data,
            gene_column_name=gene_column,
            group_column_name=group_column,
            q_value_threshold=q_value_threshold,
            logger=logger,
        )
        if zscore_filtered_feature_names:
            logger.info(
                f"Z-score filtering kept {len(zscore_filtered_feature_names)} features across "
                f"{len(significant_group_df)} significant groups."
            )
            filtered_feature_names = zscore_filtered_feature_names
        else:
            logger.warning(
                "Z-score filtering returned no significant groups; falling back to t-test filtered features."
            )
    except Exception as exc:  # defensive fallback
        logger.warning(f"In-memory z-score filtering failed; falling back to t-test features: {exc}")

    group_feature_mappings = run_grouping(
        grouping_data,
        gene_column_name=gene_column,
        group_column_name=group_column,
        filtered_features=filtered_feature_names,
        logger=logger,
    )

    # Group Scoring
    scoring_results = run_scoring(
        data_x=train_test_split_data.X_train,
        labels=train_test_split_data.y_train,
        model_name=scoring_model,
        groups=group_feature_mappings,
        output_dir=output_dir,
        iteration=iteration,
        logger=logger,
        cross_validation_folds=cross_validation_folds,
        should_save_group_features=save_group_derived_features,
        random_state=iteration_seed,
    )

    ranked_groups = scoring_results.ranked_groups
    modeling_result_list: List[ModelingResult] = []

    if not ranked_groups:
        logger.warning("No ranked groups available for modeling")
        return modeling_result_list, ranked_groups

    max_group_count = min(best_groups_to_keep, len(ranked_groups))
    previous_feature_count = 0
    for requested_groups in range(1, max_group_count + 1):
        selection = expand_groups_until_feature_increase(
            requested_top_groups=requested_groups,
            group_ranks=ranked_groups,
            group_feature_mapping=group_feature_mappings,
            data_columns=train_test_split_data.X_train.columns,
            baseline_feature_count=previous_feature_count,
            logger=logger,
        )

        step_label = f"Step {requested_groups}/{max_group_count}"
        if selection.used_top_groups == selection.requested_top_groups:
            logger.debug(
                f"{step_label}: using top {selection.used_top_groups} groups. "
                f"Unique features: {selection.baseline_feature_count} -> {selection.used_feature_count}."
            )
        elif selection.used_feature_count > selection.baseline_feature_count:
            logger.debug(
                f"{step_label}: top {selection.requested_top_groups} groups added no new features "
                f"(still {selection.baseline_feature_count}). Expanded to top "
                f"{selection.used_top_groups} groups to reach {selection.used_feature_count} unique features."
            )
        else:
            logger.debug(
                f"{step_label}: top {selection.requested_top_groups} groups added no new features "
                f"(still {selection.baseline_feature_count}). Even after expanding to "
                f"{selection.used_top_groups} groups, the feature count stayed the same."
            )

        modeling_result = run_modeling(
            data_train_x=train_test_split_data.X_train,
            data_train_y=train_test_split_data.y_train,
            data_test_x=train_test_split_data.X_test,
            data_test_y=train_test_split_data.y_test,
            group_ranks=ranked_groups,
            group_feature_mapping=group_feature_mappings,
            model_name=model_name,
            top_n_groups=selection.used_top_groups,
            logger=logger,
            random_state=iteration_seed,
        )
        modeling_result_list.append(modeling_result)

        if modeling_result.num_features_used > previous_feature_count:
            previous_feature_count = modeling_result.num_features_used

    if modeling_result_list:
        best = max(modeling_result_list, key=lambda r: r.f1_score)
        logger.info(
            f"✅ Modeling completed: {len(modeling_result_list)} steps | "
            f"Best F1={best.f1_score:.4f} (top {best.num_groups_used} groups, "
            f"{best.num_features_used} features)"
        )
    else:
        logger.info("Modeling completed (no results)")
    return modeling_result_list, ranked_groups


# -----------------------------------------------------------------------------
# Tam akış
# -----------------------------------------------------------------------------

def gsm_run(
    input_data: pd.DataFrame,
    group_data: pd.DataFrame,
    *,
    sample_ratio: float = config.TRAIN_TEST_SPLIT_RATIO,
    n_iterations: int = config.NUMBER_OF_ITERATIONS,
    model_name: str = config.MODEL_NAME,
    label_column: str = config.LABEL_COLUMN_NAME,
    positive_class_label: str = config.CLASS_LABELS_POSITIVE,
    negative_class_label: str = config.CLASS_LABELS_NEGATIVE,
    gene_column: str = config.GENE_COLUMN_NAME,
    group_column: str = config.GROUP_COLUMN_NAME,
    normalization_method: str = config.NORMALIZATION_METHOD,
    initial_feature_filter_size: int = config.INITIAL_FEATURE_FILTER_SIZE,
    initial_seed: int = config.RANDOM_SEED,
    ttest_threshold: float = config.TTEST_THRESHOLD,
    cross_validation_folds: int = config.CROSS_VALIDATION_FOLDS,
    best_groups_to_keep: int = config.BEST_GROUPS_TO_KEEP,
    save_intermediate_results: bool = config.SAVE_INTERMEDIATE_RESULTS,
    apply_class_balancing: bool = config.APPLY_CLASS_BALANCING,
    min_class_balance_ratio: float = config.MIN_CLASS_BALANCE_RATIO,
    sampling_method: str = config.SAMPLING_METHOD,
    scoring_model: str = config.SCORING_MODEL,
    run_biological_validation_flag: bool = config.RUN_BIOLOGICAL_VALIDATION,
    biological_validation_top_genes: int = config.BIOLOGICAL_VALIDATION_TOP_GENES,
    disgenet_api_key: str = config.DISGENET_API_KEY,
    logger_path: Optional[Path] = None,
    notebook_mode: bool = False,
    extra_handlers: Optional[List[logging.Handler]] = None,
    input_data_name: Optional[str] = None,
    group_data_name: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    progress_file: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> Path:
    """GSM_workflow.gsm_run klonu; z-score entegrasyonu ile."""

    project_root = Path(__file__).resolve().parents[2]

    main_data_stem = input_data_name if input_data_name else Path(config.INPUT_EXPRESSION_DATA).stem
    group_data_stem = group_data_name if group_data_name else Path(config.INPUT_GROUP_DATA).stem

    _safe_run_name = ""
    if run_name:
        import re as _re
        _safe_run_name = _re.sub(r"[^\w\-]", "_", run_name.strip())[:60]

    _folder_suffix = f"_{_safe_run_name}" if _safe_run_name else ""
    output_folder_path = Path(config.OUTPUT_DIR) / f"gsm_{time.strftime('%Y_%m_%d-%H_%M_%S')}_{main_data_stem}_{group_data_stem}{_folder_suffix}"
    output_folder_path.mkdir(parents=True, exist_ok=True)
    if logger_path is None:
        logger_path = output_folder_path / "gsm_workflow.log"

    logger = setup_logger(str(logger_path), logger_name="GSM_workflow_logger_zscore")

    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")

    runtime_params = {
        "run_name": run_name or "",
        "input_data": main_data_stem,
        "input_shape": list(input_data.shape),
        "grouping_data": group_data_stem,
        "grouping_shape": list(group_data.shape),
        "n_iterations": n_iterations,
        "sample_ratio": sample_ratio,
        "model_name": model_name,
        "label_column": label_column,
        "positive_class_label": positive_class_label,
        "negative_class_label": negative_class_label,
        "gene_column": gene_column,
        "group_column": group_column,
        "normalization_method": normalization_method,
        "initial_feature_filter_size": initial_feature_filter_size,
        "initial_seed": initial_seed,
        "ttest_threshold": ttest_threshold,
        "cross_validation_folds": cross_validation_folds,
        "best_groups_to_keep": best_groups_to_keep,
        "save_intermediate_results": save_intermediate_results,
        "apply_class_balancing": apply_class_balancing,
        "min_class_balance_ratio": min_class_balance_ratio,
        "sampling_method": sampling_method,
        "scoring_model": scoring_model,
        "run_biological_validation": run_biological_validation_flag,
        "biological_validation_top_genes": biological_validation_top_genes,
        "q_value_threshold": config.Q_VALUE_THRESHOLD,
    }

    logger.info(
        f"GSM Pipeline (z-score) | data={main_data_stem} {input_data.shape} | "
        f"groups={group_data_stem} {group_data.shape} | "
        f"iters={n_iterations} | model={model_name}"
    )
    log_runtime_config(runtime_params=runtime_params, logger=logger)
    save_runtime_config(output_dir=output_folder_path, runtime_params=runtime_params, logger=logger)

    if extra_handlers:
        for handler in extra_handlers:
            logger.addHandler(handler)

    logger.info("🚀 Starting GSM pipeline (z-score variant)")

    if progress_file is not None:
        try:
            _write_progress(progress_file, phase="preprocessing", iteration=0, total=n_iterations)
        except Exception:
            pass

    # Preprocess data
    logger.info("Preprocessing data...")
    data_preprocessed = preprocess_data(
        input_data,
        label_column,
        logger=logger,
        label_of_negative_class=negative_class_label,
        label_of_positive_class=positive_class_label,
        normalization_method=normalization_method,
        apply_class_balancing=apply_class_balancing,
        min_class_balance_ratio=min_class_balance_ratio,
        sampling_method=sampling_method,
    )
    logger.info("Data preprocessed.")

    inference_scaler = fit_scaler(
        data_preprocessed,
        label_column_name=label_column,
        method=normalization_method,
    )
    logger.debug("Normalization scaler fitted for inference bundle")

    group_data_processed = preprocess_grouping_data(
        group_data,
        gene_column_name=gene_column,
        group_column_name=group_column,
        logger=logger,
    )
    logger.info("Grouping data validated.")

    # One-time feature scoring (rapor amaçlı)
    set_random_seed(initial_seed)
    logger.info("Feature scoring (one-time)...")
    X_full = data_preprocessed.drop(columns=[label_column])
    y_full = data_preprocessed[label_column]
    feature_scores = score_all_features(X_full, y_full, logger, random_state=initial_seed)
    features_output = output_folder_path / "individual_feature_scores.csv"
    save_ranked_features(
        FeatureRankingOutput(
            output_path=features_output,
            feature_scores=feature_scores,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            model_name=model_name,
            iteration=0,
        ),
        logger,
    )

    iteration_results: List = []
    iteration_times: List[float] = []
    collected_model_artifacts: List[ModelArtifact] = []
    pipeline_start_time = time.time()

    for i in range(1, n_iterations + 1):
        iteration_start_time = time.time()
        iteration_seed = generate_iteration_seed(initial_seed, i)
        set_random_seed(iteration_seed)

        logger.info("=" * 50)
        logger.info(f"Iteration {i}/{n_iterations} (seed={iteration_seed})")

        modeling_results, ranked_groups = gsm_main_loop_zscore(
            data=data_preprocessed,
            grouping_data=group_data_processed,
            model_name=model_name,
            scoring_model=scoring_model,
            output_dir=output_folder_path,
            iteration=i,
            logger=logger,
            gene_column=gene_column,
            group_column=group_column,
            label_column=label_column,
            sample_ratio=sample_ratio,
            ttest_threshold=ttest_threshold,
            q_value_threshold=config.Q_VALUE_THRESHOLD,
            initial_feature_filter_size=initial_feature_filter_size,
            best_groups_to_keep=best_groups_to_keep,
            cross_validation_folds=cross_validation_folds,
            iteration_seed=iteration_seed,
            save_group_derived_features=(i == 1),
        )

        iteration_results.append(
            {
                "iteration": i,
                "random_seed": iteration_seed,
                "modeling_results": modeling_results,
                "ranked_groups": ranked_groups,
            }
        )

        if modeling_results:
            best_model_result = max(modeling_results, key=lambda r: r.f1_score)
            if best_model_result.fitted_model is not None:
                collected_model_artifacts.append(
                    ModelArtifact(
                        model=best_model_result.fitted_model,
                        iteration=i,
                        f1_score=best_model_result.f1_score,
                        auc_roc=best_model_result.auc_roc,
                        num_features_used=best_model_result.num_features_used,
                        num_groups_used=best_model_result.num_groups_used,
                    )
                )

        iteration_duration = time.time() - iteration_start_time
        iteration_times.append(iteration_duration)

        avg_time = sum(iteration_times) / len(iteration_times)
        remaining = n_iterations - i
        eta_seconds = avg_time * remaining
        elapsed_total = time.time() - pipeline_start_time
        estimated_finish_time = datetime.now() + timedelta(seconds=eta_seconds)

        if remaining > 0:
            logger.info(
                f"Iter {i}/{n_iterations} done in {format_duration(iteration_duration)} | "
                f"Elapsed {format_duration(elapsed_total)} | "
                f"ETA {estimated_finish_time.strftime('%H:%M:%S')} (~{format_duration(eta_seconds)} left)"
            )
        else:
            logger.info(f"Iter {i}/{n_iterations} done in {format_duration(iteration_duration)}")

        if progress_callback is not None:
            try:
                progress_callback(i, n_iterations, elapsed_total, eta_seconds)
            except Exception:
                pass
        if progress_file is not None:
            try:
                _write_progress(
                    progress_file,
                    phase="iteration",
                    iteration=i,
                    total=n_iterations,
                    elapsed=elapsed_total,
                    eta_seconds=eta_seconds,
                )
            except Exception:
                pass

    # Post-processing
    if progress_file is not None:
        try:
            total_elapsed = time.time() - pipeline_start_time
            _write_progress(
                progress_file,
                phase="post-processing",
                iteration=n_iterations,
                total=n_iterations,
                elapsed=total_elapsed,
                eta_seconds=0,
            )
        except Exception:
            pass
    if progress_callback is not None:
        try:
            total_elapsed = time.time() - pipeline_start_time
            progress_callback(n_iterations, n_iterations, total_elapsed, 0)
        except Exception:
            pass

    if save_intermediate_results:
        logger.info("Saving results...")
        save_results.save_modeling_results(
            results=[r["modeling_results"] for r in iteration_results],
            iteration_metadata=[
                save_results.IterationMetadata(iteration=r["iteration"], random_seed=r["random_seed"])
                for r in iteration_results
            ],
            output_dir=str(output_folder_path),
            experiment_name="modeling_results",
            logger=logger,
        )

    logger.info("Generating visualizations...")
    viz_data = []
    for res in iteration_results:
        for model_res in res["modeling_results"]:
            viz_data.append(
                {
                    "Iteration": res["iteration"],
                    "NumGroups": model_res.num_groups_used,
                    "F1Score": model_res.f1_score,
                }
            )
    if viz_data:
        viz_df = pd.DataFrame(viz_data)
        visualize_f1_scores(viz_df, output_folder_path, logger)
    else:
        logger.warning("No data available for visualization")

    results_json_path = output_folder_path / "modeling_results_all_iterations.json"
    postprocessing_results = run_postprocessing_aggregations(iteration_results, output_folder_path, logger)
    all_results_data = postprocessing_results["all_results_data"]
    aggregated_groups = postprocessing_results["aggregated_groups"]
    aggregated_features = postprocessing_results["aggregated_features"]
    robust_rank_groups = postprocessing_results["robust_rank_groups"]
    robust_rank_features = postprocessing_results["robust_rank_features"]

    logger.info("Generating summary report...")
    save_results.save_summary_report(
        results=[r["modeling_results"] for r in iteration_results],
        iteration_metadata=[
            save_results.IterationMetadata(iteration=r["iteration"], random_seed=r["random_seed"])
            for r in iteration_results
        ],
        output_dir=output_folder_path,
        logger=logger,
        aggregated_groups=aggregated_groups,
        aggregated_features=aggregated_features,
        robust_rank_groups=robust_rank_groups,
        robust_rank_features=robust_rank_features,
    )

    if not results_json_path.exists():
        ensure_results_json(results_json_path, all_results_data, logger)

    try:
        generate_all_figures(output_folder_path, results_json_path, logger)
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")

    if run_biological_validation_flag and results_json_path.exists():
        try:
            # Import biological validation lazily so optional API dependencies do not block normal workflow runs.
            from src.utils.biological_validation import run_biological_validation

            logger.info("🧬 Running biological validation...")
            grouping_data_path = project_root / config.INPUT_GROUP_DATA
            run_biological_validation(
                output_dir=output_folder_path,
                results_json_path=results_json_path,
                logger=logger,
                disgenet_api_key=disgenet_api_key or None,
                top_n_genes=biological_validation_top_genes,
                grouping_data_path=grouping_data_path if grouping_data_path.exists() else None,
                gene_column=gene_column,
                group_column=group_column,
            )
            logger.info("✅ Biological validation completed")
        except Exception as e:
            logger.warning(f"⚠️ Biological validation failed: {e}")
            logger.warning(
                "💡 You can re-run it later with:  "
                f"python -m gsm bio-validate --run {output_folder_path}"
            )
    elif run_biological_validation_flag:
        logger.warning("Biological validation skipped: results JSON not found")

    if collected_model_artifacts:
        try:
            best_overall = max(collected_model_artifacts, key=lambda m: m.f1_score)
            best_features: List[str] = []
            best_groups: List[str] = []
            for res in iteration_results:
                for mr in res["modeling_results"]:
                    if mr.f1_score == best_overall.f1_score and mr.used_features:
                        best_features = mr.used_features
                        best_groups = mr.used_groups
                        break
                if best_features:
                    break

            if best_features:
                bundle_path = save_model_bundle(
                    models=collected_model_artifacts,
                    feature_names=best_features,
                    group_names=best_groups,
                    scaler=inference_scaler,
                    normalization_method=normalization_method,
                    label_mapping={"positive": positive_class_label, "negative": negative_class_label},
                    dataset_name=main_data_stem,
                    model_name=model_name,
                    n_training_samples=len(data_preprocessed),
                    n_iterations_total=n_iterations,
                    random_seed=initial_seed,
                    output_dir=output_folder_path,
                    logger=logger,
                    run_name=run_name or "",
                )
                logger.info(f"🏅 Clinical inference bundle: {bundle_path.name}")
            else:
                logger.warning("No feature names found; skipping bundle save")
        except Exception as e:
            logger.warning(f"Model bundle save failed: {e}")
    else:
        logger.warning("No models collected; skipping bundle save")

    logger.info("✅ GSM pipeline completed successfully (z-score variant)")
    return output_folder_path


# -----------------------------------------------------------------------------
# CLI girişi
# -----------------------------------------------------------------------------

def main() -> None:
    project_folder = Path().resolve()
    print(f"[info] Project Folder: {project_folder}")

    input_file = project_folder / config.INPUT_EXPRESSION_DATA
    group_file = project_folder / config.INPUT_GROUP_DATA

    print(f"[info] Loading expression data from: {input_file}")
    print(f"[info] Loading group data from: {group_file}")

    input_data = load_input_file(input_file, separator=config.MAIN_DATA_FILE_SEPARATOR)
    group_data = load_group_file(group_file, separator=config.GROUPING_FILE_SEPARATOR)

    print(f"[done] Expression data loaded: {input_data.shape}")
    print(f"[done] Group data loaded: {group_data.shape}")

    print("[run] Starting GSM z-score workflow execution...")
    gsm_run(input_data, group_data)
    print("[done] GSM z-score workflow completed.")


if __name__ == "__main__":
    main()
