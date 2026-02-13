"""
🧪 Unit Tests for Core GSM Pipeline Functions

Purpose:
    Targeted tests for the most critical data processing and scoring functions.
    Uses the test data in data/test/ for realistic validation.

Run:
    python -m pytest tests/test_core_functions.py -v
    
    Or simply:
    python tests/test_core_functions.py
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest


##### Test Logger (shared across tests) #####
_test_logger = logging.getLogger("test_logger")
_test_logger.setLevel(logging.WARNING)
if not _test_logger.handlers:
    _test_logger.addHandler(logging.StreamHandler())


##### Test Data Helpers #####
def make_expression_data(n_samples: int = 40, n_features: int = 10) -> pd.DataFrame:
    """Create a small synthetic expression dataset with binary class labels."""
    np.random.seed(42)
    features = {f"gene_{i}": np.random.randn(n_samples) for i in range(n_features)}
    labels = ["pos"] * (n_samples // 2) + ["neg"] * (n_samples // 2)
    features["class"] = labels
    return pd.DataFrame(features)


def make_imbalanced_data(n_majority: int = 30, n_minority: int = 10) -> pd.DataFrame:
    """Create a dataset with imbalanced classes."""
    np.random.seed(42)
    n_total = n_majority + n_minority
    features = {f"gene_{i}": np.random.randn(n_total) for i in range(5)}
    labels = ["pos"] * n_majority + ["neg"] * n_minority
    features["class"] = labels
    return pd.DataFrame(features)


def make_grouping_data() -> pd.DataFrame:
    """Create a small grouping dataset matching the synthetic expression data."""
    return pd.DataFrame({
        "feature_id": ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4",
                        "gene_5", "gene_6", "gene_7", "gene_0", "gene_1"],
        "group_name": ["group_A", "group_A", "group_A", "group_B", "group_B",
                        "group_B", "group_C", "group_C", "group_C", "group_B"]
    })


##### 1. Data Preprocessing Tests #####
class TestDataPreprocessing:
    """Tests for data_preprocess.py functions."""

    def test_validate_input_data_valid(self):
        """Valid data should pass validation without errors."""
        from src.data_processing.data_preprocess import validate_input_data
        data = make_expression_data()
        validate_input_data(data, "class")

    def test_validate_input_data_empty(self):
        """Empty dataframe should raise ValueError."""
        from src.data_processing.data_preprocess import validate_input_data
        with pytest.raises(ValueError, match="empty"):
            validate_input_data(pd.DataFrame(), "class")

    def test_validate_input_data_missing_column(self):
        """Missing label column should raise ValueError."""
        from src.data_processing.data_preprocess import validate_input_data
        data = make_expression_data().drop(columns=["class"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_input_data(data, "class")

    def test_convert_labels_to_binary(self):
        """Labels should be converted to 0 and 1."""
        from src.data_processing.data_preprocess import convert_labels_to_binary
        data = make_expression_data()
        result = convert_labels_to_binary(data, "class", "neg", "pos")
        assert set(result["class"].unique()) == {0, 1}
        assert result["class"].sum() == 20  # 20 positive samples

    def test_convert_labels_invalid(self):
        """Invalid labels should raise ValueError."""
        from src.data_processing.data_preprocess import convert_labels_to_binary
        data = make_expression_data()
        data.loc[0, "class"] = "unknown"
        with pytest.raises(ValueError, match="Invalid labels"):
            convert_labels_to_binary(data, "class", "neg", "pos")

    def test_preprocess_data_shape(self):
        """Preprocessed data should have same number of columns as input."""
        from src.data_processing.data_preprocess import preprocess_data
        data = make_expression_data()
        result = preprocess_data(
            data, "class",
            label_of_negative_class="neg",
            label_of_positive_class="pos",
            logger=_test_logger
        )
        # Should have same columns (features + class label)
        assert result.shape[1] == data.shape[1]
        # Binary labels
        assert set(result["class"].unique()) == {0, 1}

    def test_preprocess_grouping_data_valid(self):
        """Valid grouping data should pass through correctly."""
        from src.data_processing.data_preprocess import preprocess_grouping_data
        data = make_grouping_data()
        result = preprocess_grouping_data(data, logger=_test_logger)
        assert len(result) == 10
        assert "feature_id" in result.columns
        assert "group_name" in result.columns

    def test_preprocess_grouping_data_missing_column(self):
        """Missing required column should raise ValueError."""
        from src.data_processing.data_preprocess import preprocess_grouping_data
        data = pd.DataFrame({"wrong_col": ["a", "b"]})
        with pytest.raises(ValueError):
            preprocess_grouping_data(data, logger=_test_logger)


##### 2. Class Balancing Tests #####
class TestClassBalancing:
    """Tests for class balancing in data_preprocess.py."""

    def test_undersampling_reduces_majority(self):
        """Undersampling should reduce majority class to match minority."""
        from src.data_processing.data_preprocess import determine_class_balance
        data = make_imbalanced_data(n_majority=30, n_minority=10)
        data = data.copy()
        data["class"] = data["class"].map({"pos": 1, "neg": 0})

        result = determine_class_balance(
            data, logger=_test_logger,
            label_column_name="class",
            sampling_method="undersampling"
        )
        counts = result["class"].value_counts()
        assert counts[0] == counts[1], f"Classes not balanced: {counts.to_dict()}"
        assert len(result) == 20  # 10 + 10

    def test_oversampling_increases_minority(self):
        """Oversampling should increase minority class size."""
        from src.data_processing.data_preprocess import determine_class_balance
        data = make_imbalanced_data(n_majority=30, n_minority=10)
        data = data.copy()
        data["class"] = data["class"].map({"pos": 1, "neg": 0})

        result = determine_class_balance(
            data, logger=_test_logger,
            label_column_name="class",
            sampling_method="oversampling"
        )
        # Minority class should have grown
        assert len(result) > 40

    def test_balanced_data_unchanged(self):
        """Already balanced data should not be modified."""
        from src.data_processing.data_preprocess import determine_class_balance
        data = make_expression_data()  # 20 pos + 20 neg = balanced
        data = data.copy()
        data["class"] = data["class"].map({"pos": 1, "neg": 0})

        result = determine_class_balance(
            data, logger=_test_logger,
            label_column_name="class"
        )
        assert len(result) == 40  # Unchanged

    def test_preprocess_with_balancing_enabled(self):
        """Full preprocessing with class balancing should balance classes."""
        from src.data_processing.data_preprocess import preprocess_data
        data = make_imbalanced_data(n_majority=30, n_minority=10)
        result = preprocess_data(
            data, "class",
            label_of_negative_class="neg",
            label_of_positive_class="pos",
            logger=_test_logger,
            apply_class_balancing=True,
            sampling_method="undersampling"
        )
        counts = result["class"].value_counts()
        assert counts[0] == counts[1]


##### 3. Train/Test Splitting Tests #####
class TestTrainTestSplit:
    """Tests for train_test_splitter.py."""

    def test_basic_split(self):
        """Basic split should produce correct train/test proportions."""
        from src.data_processing.train_test_splitter import split_data
        data = make_expression_data()
        data["class"] = data["class"].map({"pos": 1, "neg": 0})
        result = split_data(data, "class", test_size=0.3)

        expected_test = int(40 * 0.3)
        assert abs(len(result.X_test) - expected_test) <= 1
        assert len(result.X_train) + len(result.X_test) == 40

    def test_stratified_split(self):
        """Stratified split should maintain class distribution."""
        from src.data_processing.train_test_splitter import split_data
        data = make_expression_data()
        data["class"] = data["class"].map({"pos": 1, "neg": 0})
        result = split_data(data, "class", test_size=0.3, stratify=True)

        # Check class balance is roughly maintained in test set
        test_ratio = result.y_test.mean()
        assert 0.3 < test_ratio < 0.7, f"Test set imbalanced: {test_ratio}"

    def test_reproducible_split(self):
        """Same seed should produce identical splits."""
        from src.data_processing.train_test_splitter import split_data
        data = make_expression_data()
        data["class"] = data["class"].map({"pos": 1, "neg": 0})

        r1 = split_data(data, "class", test_size=0.3, random_state=42)
        r2 = split_data(data, "class", test_size=0.3, random_state=42)
        pd.testing.assert_frame_equal(r1.X_train, r2.X_train)

    def test_invalid_test_size(self):
        """Invalid test_size should raise ValueError."""
        from src.data_processing.train_test_splitter import split_data
        data = make_expression_data()
        data["class"] = data["class"].map({"pos": 1, "neg": 0})
        with pytest.raises(ValueError):
            split_data(data, "class", test_size=1.5)


##### 4. Scoring Tests #####
class TestScoring:
    """Tests for scoring functions."""

    def test_score_data_returns_metrics(self):
        """score_data should return MetricsData with expected fields."""
        from src.scoring.score_data import ScoringParameters, score_data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
        y = pd.Series([0] * 25 + [1] * 25)

        params = ScoringParameters(
            data_x=X, group_name="test_group",
            labels=y, classifier_name="DecisionTree",
            cross_validation_folds=3, logger=_test_logger
        )
        result = score_data(params)

        assert result.accuracy is not None
        assert result.f1 is not None
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.f1 <= 1

    def test_score_data_empty_raises(self):
        """Empty data should raise ValueError."""
        from src.scoring.score_data import ScoringParameters, score_data
        X = pd.DataFrame()
        y = pd.Series(dtype=int)

        params = ScoringParameters(
            data_x=X, group_name="empty",
            labels=y, classifier_name="DecisionTree",
            cross_validation_folds=3, logger=_test_logger
        )
        with pytest.raises(ValueError):
            score_data(params)


##### 5. Classification Model Tests #####
class TestClassificationModels:
    """Tests for model creation in classification.py."""

    def test_get_random_forest(self):
        """RandomForest classifier should be created correctly."""
        from src.machine_learning.classification import get_classifier_object
        clf = get_classifier_object("RandomForest")
        assert clf is not None
        assert hasattr(clf, "fit")

    def test_get_decision_tree(self):
        """DecisionTree classifier should be created correctly."""
        from src.machine_learning.classification import get_classifier_object
        clf = get_classifier_object("DecisionTree")
        assert clf is not None
        assert hasattr(clf, "fit")

    def test_unsupported_model_raises(self):
        """Unsupported model name should raise ValueError."""
        from src.machine_learning.classification import get_classifier_object
        with pytest.raises(ValueError):
            get_classifier_object("NonExistentModel")


##### 6. Feature Scoring Tests #####
class TestFeatureScoring:
    """Tests for feature_scorer.py."""

    def test_feature_score_validation(self):
        """FeatureScore should validate score ranges."""
        from src.scoring.feature_scorer import FeatureScore
        score = FeatureScore(
            feature_name="gene_0",
            f1_score=0.8,
            importance_score=0.5,
            mutual_info=0.3
        )
        assert score.feature_name == "gene_0"

    def test_feature_score_invalid_f1(self):
        """F1 score out of range should raise ValueError."""
        from src.scoring.feature_scorer import FeatureScore
        with pytest.raises(ValueError):
            FeatureScore(feature_name="x", f1_score=1.5,
                        importance_score=0.5, mutual_info=0.1)

    def test_feature_score_negative_mi(self):
        """Negative mutual info should raise ValueError."""
        from src.scoring.feature_scorer import FeatureScore
        with pytest.raises(ValueError):
            FeatureScore(feature_name="x", f1_score=0.5,
                        importance_score=0.5, mutual_info=-0.1)


##### 7. Config Tests #####
class TestConfig:
    """Tests for GSM_workflow_config.py consistency."""

    def test_config_values_reasonable(self):
        """Config values should be within reasonable ranges."""
        from src.workflows.GSM_workflow_config import (
            TRAIN_TEST_SPLIT_RATIO, TTEST_THRESHOLD,
            CROSS_VALIDATION_FOLDS, BEST_GROUPS_TO_KEEP,
            NUMBER_OF_ITERATIONS, RANDOM_SEED
        )
        assert 0.5 <= TRAIN_TEST_SPLIT_RATIO <= 0.9
        assert 0.001 <= TTEST_THRESHOLD <= 0.1
        assert 2 <= CROSS_VALIDATION_FOLDS <= 10
        assert 1 <= BEST_GROUPS_TO_KEEP <= 100
        assert 1 <= NUMBER_OF_ITERATIONS <= 1000
        assert isinstance(RANDOM_SEED, int)

    def test_scoring_model_valid(self):
        """Scoring model should be a valid option."""
        from src.workflows.GSM_workflow_config import SCORING_MODEL
        assert SCORING_MODEL in ("DecisionTree", "RandomForest", "XGBoost")

    def test_model_name_valid(self):
        """Model name should be a valid option."""
        from src.workflows.GSM_workflow_config import MODEL_NAME
        assert MODEL_NAME in ("DecisionTree", "RandomForest", "XGBoost", "SVM", "KNN", "MLP")


##### 8. Seed Reproducibility Tests #####
class TestReproducibility:
    """Tests for seed generation and reproducibility."""

    def test_generate_iteration_seed_deterministic(self):
        """Same inputs should produce same seed."""
        from src.workflows.GSM_workflow import generate_iteration_seed
        s1 = generate_iteration_seed(44, 1)
        s2 = generate_iteration_seed(44, 1)
        assert s1 == s2

    def test_generate_iteration_seed_varies(self):
        """Different iterations should produce different seeds."""
        from src.workflows.GSM_workflow import generate_iteration_seed
        s1 = generate_iteration_seed(44, 1)
        s2 = generate_iteration_seed(44, 2)
        assert s1 != s2


##### Run tests directly #####
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
