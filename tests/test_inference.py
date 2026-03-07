"""
🧪 Tests for Clinical Inference Pipeline

Purpose:
    Test the model bundle save/load round-trip, inference engine,
    and clinical report generation using synthetic data.

Run:
    python -m pytest tests/test_inference.py -v
"""

import sys
import logging
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


##### Test Logger #####
_test_logger = logging.getLogger("test_inference")
_test_logger.setLevel(logging.INFO)
if not _test_logger.handlers:
    _test_logger.addHandler(logging.StreamHandler())


##### Fixtures #####

@pytest.fixture
def synthetic_training_data():
    """Create synthetic expression data for training."""
    np.random.seed(42)
    n_samples, n_features = 60, 20
    X = np.random.randn(n_samples, n_features)
    # Make first 5 features predictive
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    feature_names = [f"gene_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="class")


@pytest.fixture
def trained_model_artifacts(synthetic_training_data):
    """Train multiple models and return ModelArtifact list."""
    from src.inference.model_bundle import ModelArtifact

    X, y = synthetic_training_data
    artifacts = []

    for i in range(5):
        # Train with different random states to simulate iterations
        model = RandomForestClassifier(n_estimators=10, random_state=42 + i)
        model.fit(X, y)
        y_pred = model.predict(X)
        f1 = float(np.mean(y_pred == y))  # Simplified metric

        artifacts.append(ModelArtifact(
            model=model,
            iteration=i + 1,
            f1_score=f1,
            auc_roc=f1 * 0.95,
            num_features_used=20,
            num_groups_used=3,
        ))

    return artifacts


@pytest.fixture
def fitted_scaler(synthetic_training_data):
    """Fit a StandardScaler on synthetic data."""
    X, _ = synthetic_training_data
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


@pytest.fixture
def feature_names():
    """Feature names matching synthetic data."""
    return [f"gene_{i}" for i in range(20)]


##### Tests: Model Bundle Save/Load #####

class TestModelBundle:
    """Tests for model bundle save and load operations."""

    def test_save_bundle_creates_file(
        self, trained_model_artifacts, fitted_scaler, feature_names
    ):
        """save_bundle() should create a .gsm.zip file."""
        from src.inference.model_bundle import save_bundle

        with tempfile.TemporaryDirectory() as tmp:
            path = save_bundle(
                models=trained_model_artifacts,
                feature_names=feature_names,
                group_names=["cancer_group_A", "cancer_group_B", "cancer_group_C"],
                scaler=fitted_scaler,
                normalization_method="zscore",
                label_mapping={"positive": "pos", "negative": "neg"},
                dataset_name="test_dataset",
                model_name="RandomForest",
                n_training_samples=60,
                n_iterations_total=5,
                random_seed=42,
                output_dir=Path(tmp),
                logger=_test_logger,
                max_models=3,
            )

            assert path.exists(), "Bundle file should exist"
            assert path.name.endswith(".gsm.zip"), "Bundle should have .gsm.zip extension"
            assert path.stat().st_size > 0, "Bundle should not be empty"

    def test_bundle_round_trip(
        self, trained_model_artifacts, fitted_scaler, feature_names
    ):
        """Save then load should produce identical metadata."""
        from src.inference.model_bundle import save_bundle, load_bundle

        with tempfile.TemporaryDirectory() as tmp:
            path = save_bundle(
                models=trained_model_artifacts,
                feature_names=feature_names,
                group_names=["group_A", "group_B"],
                scaler=fitted_scaler,
                normalization_method="zscore",
                label_mapping={"positive": "disease", "negative": "control"},
                dataset_name="GDS_test",
                model_name="RandomForest",
                n_training_samples=60,
                n_iterations_total=5,
                random_seed=42,
                output_dir=Path(tmp),
                logger=_test_logger,
                max_models=3,
            )

            bundle = load_bundle(path, logger=_test_logger)

            assert bundle.metadata.dataset_name == "GDS_test"
            assert bundle.metadata.n_models_saved == 3
            assert bundle.metadata.n_features == 20
            assert bundle.metadata.normalization_method == "zscore"
            assert bundle.metadata.label_mapping["positive"] == "disease"
            assert len(bundle.models) == 3
            assert bundle.scaler is not None

    def test_bundle_info_readable(
        self, trained_model_artifacts, fitted_scaler, feature_names
    ):
        """bundle_info() should return a non-empty string."""
        from src.inference.model_bundle import save_bundle, bundle_info

        with tempfile.TemporaryDirectory() as tmp:
            path = save_bundle(
                models=trained_model_artifacts,
                feature_names=feature_names,
                group_names=["group_A"],
                scaler=fitted_scaler,
                normalization_method="zscore",
                label_mapping={"positive": "pos", "negative": "neg"},
                dataset_name="test",
                model_name="RandomForest",
                n_training_samples=60,
                n_iterations_total=5,
                random_seed=42,
                output_dir=Path(tmp),
                logger=_test_logger,
            )

            info = bundle_info(path)
            assert "Bundle ID" in info
            assert "RandomForest" in info
            assert "test" in info


##### Tests: Inference Engine #####

class TestInferenceEngine:
    """Tests for the inference engine."""

    def _make_bundle(self, trained_model_artifacts, fitted_scaler, feature_names):
        """Helper to create a bundle for testing."""
        from src.inference.model_bundle import save_bundle, load_bundle

        with tempfile.TemporaryDirectory() as tmp:
            path = save_bundle(
                models=trained_model_artifacts,
                feature_names=feature_names,
                group_names=["group_A"],
                scaler=fitted_scaler,
                normalization_method="zscore",
                label_mapping={"positive": "disease", "negative": "control"},
                dataset_name="test",
                model_name="RandomForest",
                n_training_samples=60,
                n_iterations_total=5,
                random_seed=42,
                output_dir=Path(tmp),
                logger=_test_logger,
                max_models=3,
            )
            return load_bundle(path, logger=_test_logger)

    def test_infer_produces_results(
        self, trained_model_artifacts, fitted_scaler, feature_names
    ):
        """infer() should produce InferenceSummary with correct sample count."""
        from src.inference.inference_engine import infer

        bundle = self._make_bundle(
            trained_model_artifacts, fitted_scaler, feature_names
        )

        # Create patient data with matching features
        np.random.seed(99)
        patient_data = pd.DataFrame(
            np.random.randn(5, 20),
            columns=feature_names,
        )

        summary = infer(bundle, patient_data, logger=_test_logger)

        assert summary.n_samples == 5
        assert len(summary.results) == 5
        assert summary.n_positive + summary.n_negative == 5

    def test_infer_result_fields(
        self, trained_model_artifacts, fitted_scaler, feature_names
    ):
        """Each InferenceResult should have required fields."""
        from src.inference.inference_engine import infer

        bundle = self._make_bundle(
            trained_model_artifacts, fitted_scaler, feature_names
        )

        patient_data = pd.DataFrame(
            np.random.randn(3, 20),
            columns=feature_names,
        )

        summary = infer(bundle, patient_data, logger=_test_logger)

        for r in summary.results:
            assert r.predicted_class in ("positive", "negative")
            assert r.predicted_label in ("disease", "control")
            assert 0.0 <= r.confidence <= 1.0
            assert r.risk_level in ("HIGH", "MEDIUM", "LOW")
            assert 0.0 <= r.agreement_ratio <= 1.0
            assert len(r.individual_probabilities) == 3  # max_models=3

    def test_infer_handles_missing_features(
        self, trained_model_artifacts, fitted_scaler, feature_names
    ):
        """Inference should work when some features are missing (zero-filled)."""
        from src.inference.inference_engine import infer

        bundle = self._make_bundle(
            trained_model_artifacts, fitted_scaler, feature_names
        )

        # Only include half the features
        partial_features = feature_names[:10]
        patient_data = pd.DataFrame(
            np.random.randn(2, 10),
            columns=partial_features,
        )

        summary = infer(bundle, patient_data, logger=_test_logger)
        assert summary.n_samples == 2

    def test_infer_rejects_no_matching_features(
        self, trained_model_artifacts, fitted_scaler, feature_names
    ):
        """Inference should raise ValueError when no features match."""
        from src.inference.inference_engine import infer

        bundle = self._make_bundle(
            trained_model_artifacts, fitted_scaler, feature_names
        )

        # Completely wrong feature names
        patient_data = pd.DataFrame(
            np.random.randn(2, 5),
            columns=["wrong_A", "wrong_B", "wrong_C", "wrong_D", "wrong_E"],
        )

        with pytest.raises(ValueError, match="No matching features"):
            infer(bundle, patient_data, logger=_test_logger)


##### Tests: Clinical Report #####

class TestClinicalReport:
    """Tests for clinical report generation."""

    def test_generate_report_text(
        self, trained_model_artifacts, fitted_scaler, feature_names
    ):
        """Report should contain key sections."""
        from src.inference.model_bundle import save_bundle, load_bundle
        from src.inference.inference_engine import infer
        from src.inference.clinical_report import generate_clinical_report

        with tempfile.TemporaryDirectory() as tmp:
            path = save_bundle(
                models=trained_model_artifacts,
                feature_names=feature_names,
                group_names=["group_A"],
                scaler=fitted_scaler,
                normalization_method="zscore",
                label_mapping={"positive": "pos", "negative": "neg"},
                dataset_name="test",
                model_name="RandomForest",
                n_training_samples=60,
                n_iterations_total=5,
                random_seed=42,
                output_dir=Path(tmp),
                logger=_test_logger,
                max_models=3,
            )
            bundle = load_bundle(path, logger=_test_logger)

        patient_data = pd.DataFrame(
            np.random.randn(3, 20), columns=feature_names
        )
        summary = infer(bundle, patient_data, logger=_test_logger)
        report = generate_clinical_report(summary)

        assert "CLINICAL INFERENCE REPORT" in report
        assert "DISCLAIMER" in report
        assert "Confidence" in report

    def test_generate_report_dataframe(
        self, trained_model_artifacts, fitted_scaler, feature_names
    ):
        """Report DataFrame should have correct shape and columns."""
        from src.inference.model_bundle import save_bundle, load_bundle
        from src.inference.inference_engine import infer
        from src.inference.clinical_report import generate_report_dataframe

        with tempfile.TemporaryDirectory() as tmp:
            path = save_bundle(
                models=trained_model_artifacts,
                feature_names=feature_names,
                group_names=["group_A"],
                scaler=fitted_scaler,
                normalization_method="zscore",
                label_mapping={"positive": "pos", "negative": "neg"},
                dataset_name="test",
                model_name="RandomForest",
                n_training_samples=60,
                n_iterations_total=5,
                random_seed=42,
                output_dir=Path(tmp),
                logger=_test_logger,
                max_models=3,
            )
            bundle = load_bundle(path, logger=_test_logger)

        patient_data = pd.DataFrame(
            np.random.randn(4, 20), columns=feature_names
        )
        summary = infer(bundle, patient_data, logger=_test_logger)
        df = generate_report_dataframe(summary)

        assert len(df) == 4
        assert "Sample_ID" in df.columns
        assert "Risk_Level" in df.columns
        assert "Confidence" in df.columns
