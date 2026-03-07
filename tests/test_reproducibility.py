"""
Reproducibility Test 🔄

Purpose:
    Verify that running the GSM pipeline twice with the same parameters
    produces identical results. This catches non-determinism from:
    - Parallel joblib workers not inheriting global random seeds
    - Missing random_state on sklearn classifiers
    - Uncontrolled bootstrap/CV randomness

Test Strategy:
    1. Run the pipeline twice with identical config on test data
    2. Compare all numerical outputs (F1 scores, rankings, etc.)
    3. Both runs must produce bit-for-bit identical results
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import pytest

from src.data_processing.data_loader import load_input_file, load_group_file
from src.workflows.GSM_workflow import gsm_run


##### TEST CONFIGURATION #####
TEST_DATA_DIR = project_root / "data" / "test"
TEST_MAIN_DATA = TEST_DATA_DIR / "test_expression_data.csv"
TEST_GROUP_DATA = TEST_DATA_DIR / "test_grouping_data.csv"

# Minimal config for fast testing
TEST_ITERATIONS = 2
TEST_SEED = 44
TEST_SAMPLE_RATIO = 0.7


def _run_pipeline_once(output_label: str) -> Path:
    """Run the pipeline once and return the output directory."""
    input_data = load_input_file(TEST_MAIN_DATA, separator="auto")
    group_data = load_group_file(TEST_GROUP_DATA, separator="auto")

    output_dir = gsm_run(
        input_data,
        group_data,
        sample_ratio=TEST_SAMPLE_RATIO,
        n_iterations=TEST_ITERATIONS,
        model_name="RandomForest",
        label_column="class",
        positive_class_label="pos",
        negative_class_label="neg",
        gene_column="feature_id",
        group_column="group_name",
        normalization_method="zscore",
        initial_feature_filter_size=0,
        initial_seed=TEST_SEED,
        ttest_threshold=0.05,
        cross_validation_folds=3,
        best_groups_to_keep=3,
        save_intermediate_results=True,
        apply_class_balancing=False,
        min_class_balance_ratio=0.0,
        sampling_method="none",
        scoring_model="RandomForest",
        run_biological_validation_flag=False,
        biological_validation_top_genes=20,
        disgenet_api_key="",
        input_data_name=f"repro_test_{output_label}",
        group_data_name="test_grouping",
    )
    return output_dir


def _load_results_json(output_dir: Path) -> dict:
    """Load the main results JSON from an output directory."""
    results_path = output_dir / "modeling_results_all_iterations.json"
    assert results_path.exists(), f"Results JSON not found: {results_path}"
    with open(results_path) as f:
        return json.load(f)


def _extract_f1_scores(results: list) -> list:
    """Extract all F1 scores from results JSON for comparison."""
    f1_scores = []
    for iteration in results:
        for model_result in iteration.get("results", []):
            f1_scores.append(model_result.get("f1_score", 0.0))
    return f1_scores


def _extract_group_rankings(results: list) -> list:
    """Extract group rankings from results JSON for comparison."""
    rankings = []
    for iteration in results:
        for model_result in iteration.get("results", []):
            rankings.append(model_result.get("used_groups", []))
    return rankings


def _extract_cv_metrics(results: list) -> list:
    """Extract CV F1 means from results — sensitive to scoring randomness."""
    cv_f1 = []
    for iteration in results:
        for model_result in iteration.get("results", []):
            cv_f1.append(model_result.get("cv_f1_mean", 0.0))
    return cv_f1


@pytest.mark.slow
def test_pipeline_reproducibility():
    """Two identical pipeline runs must produce identical F1 scores and rankings."""
    # Run 1
    dir_1 = _run_pipeline_once("run1")
    results_1 = _load_results_json(dir_1)
    f1_run1 = _extract_f1_scores(results_1)
    groups_run1 = _extract_group_rankings(results_1)
    cv_run1 = _extract_cv_metrics(results_1)

    # Run 2
    dir_2 = _run_pipeline_once("run2")
    results_2 = _load_results_json(dir_2)
    f1_run2 = _extract_f1_scores(results_2)
    groups_run2 = _extract_group_rankings(results_2)
    cv_run2 = _extract_cv_metrics(results_2)

    # Compare F1 scores
    assert len(f1_run1) == len(f1_run2), (
        f"Different number of results: {len(f1_run1)} vs {len(f1_run2)}"
    )
    for i, (f1_a, f1_b) in enumerate(zip(f1_run1, f1_run2)):
        assert f1_a == f1_b, (
            f"F1 score mismatch at index {i}: {f1_a} vs {f1_b}"
        )

    # Compare group rankings
    assert groups_run1 == groups_run2, "Group rankings differ between runs"

    # Compare CV metrics (sensitive to scoring seed propagation)
    for i, (cv_a, cv_b) in enumerate(zip(cv_run1, cv_run2)):
        assert cv_a == cv_b, (
            f"CV F1 mean mismatch at index {i}: {cv_a} vs {cv_b}"
        )

    # Also compare the individual feature scores CSV (excluding timestamp column)
    feat_1 = pd.read_csv(dir_1 / "individual_feature_scores.csv")
    feat_2 = pd.read_csv(dir_2 / "individual_feature_scores.csv")
    # Drop timestamp — it will naturally differ between runs
    cols_to_compare = [c for c in feat_1.columns if c != "timestamp"]
    pd.testing.assert_frame_equal(feat_1[cols_to_compare], feat_2[cols_to_compare])

    print(f"✅ Reproducibility verified: {len(f1_run1)} scores match exactly")
    print(f"   F1 scores: {f1_run1}")
    print(f"   CV F1 means: {cv_run1}")
    print(f"   Groups: {groups_run1}")


if __name__ == "__main__":
    test_pipeline_reproducibility()
