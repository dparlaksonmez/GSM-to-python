"""
🏥 Model Bundle — Self-Contained Artifact for Clinical Inference

Purpose:
    Defines the ModelBundle dataclass and functions to save/load
    everything needed to run inference on new patient samples
    without access to the original training data or config files.

Key Functions:
    - save_bundle(): Serialize models + scaler + metadata to .gsm.zip
    - load_bundle(): Deserialize from .gsm.zip
    - bundle_info(): Human-readable summary of a bundle

File Format (.gsm.zip):
    bundle_metadata.json      — config, metrics, feature lists, creation date
    models/model_*.joblib     — fitted sklearn model objects
    preprocessing/scaler.joblib — fitted normalization scaler
    preprocessing/feature_names.json — ordered feature list

Example Usage:
    >>> bundle = save_bundle(
    ...     models=fitted_models,
    ...     feature_names=["TP53", "BRCA1", ...],
    ...     scaler=fitted_scaler,
    ...     output_dir=Path("output/bundles"),
    ...     logger=logger,
    ... )
    >>> loaded = load_bundle(Path("output/bundles/bundle_GDS2545_2026_03_06.gsm.zip"))
"""

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import sklearn


##### Data Structures #####

@dataclass
class ModelArtifact:
    """A single trained model with its iteration metadata."""
    model: Any                     # Fitted sklearn-compatible estimator
    iteration: int                 # Which GSM iteration produced this model
    f1_score: float                # F1 on held-out test set
    auc_roc: float = 0.0          # AUC-ROC on held-out test set
    num_features_used: int = 0    # Number of features the model was trained on
    num_groups_used: int = 0      # Number of disease groups contributing features


@dataclass
class BundleMetadata:
    """All metadata needed to identify and reproduce a bundle."""
    bundle_id: str
    created_at: str
    gsm_version: str
    sklearn_version: str
    # Training info
    dataset_name: str
    n_training_samples: int
    n_iterations_total: int
    n_models_saved: int
    random_seed: int
    # Feature info
    feature_names: list[str]
    group_names: list[str]
    n_features: int
    # Preprocessing
    normalization_method: str
    label_mapping: dict[str, str]
    # Performance
    ensemble_f1_mean: float
    ensemble_f1_std: float
    ensemble_auc_mean: float
    ensemble_auc_std: float
    individual_metrics: list[dict[str, float]]
    # Ensemble config
    ensemble_strategy: str
    # Model type
    model_name: str
    # Optional human-readable experiment label
    run_name: str = ""


@dataclass
class ModelBundle:
    """Self-contained artifact for clinical inference.

    Contains everything needed to preprocess new patient data
    and generate disease predictions, without access to the
    original training data or configuration files.
    """
    metadata: BundleMetadata
    models: list[ModelArtifact]
    scaler: Any                    # Fitted sklearn scaler (StandardScaler, etc.)


##### Constants #####
BUNDLE_EXTENSION = ".gsm.zip"
GSM_VERSION = "1.0.0"
DEFAULT_N_MODELS_TO_SAVE = 10
DEFAULT_ENSEMBLE_STRATEGY = "mean_probability"


##### Save Bundle #####

def save_bundle(
    *,
    models: list[ModelArtifact],
    feature_names: list[str],
    group_names: list[str],
    scaler: Any,
    normalization_method: str,
    label_mapping: dict[str, str],
    dataset_name: str,
    model_name: str,
    n_training_samples: int,
    n_iterations_total: int,
    random_seed: int,
    output_dir: Path,
    logger: logging.Logger,
    max_models: int = DEFAULT_N_MODELS_TO_SAVE,
    ensemble_strategy: str = DEFAULT_ENSEMBLE_STRATEGY,
    run_name: str = "",
) -> Path:
    """Save a ModelBundle to a compressed .gsm.zip file.

    Selects the top-K models by F1 score and packages them
    with the preprocessing scaler and full metadata.

    Args:
        models: All trained ModelArtifact objects from the pipeline
        feature_names: Ordered list of gene/feature names the models expect
        group_names: Disease group names used in feature selection
        scaler: Fitted sklearn scaler from preprocessing
        normalization_method: 'zscore', 'minmax', or 'robust'
        label_mapping: {'positive': 'pos', 'negative': 'neg'}
        dataset_name: Name of the training dataset (e.g. 'GDS2545')
        model_name: Classifier type (e.g. 'RandomForest')
        n_training_samples: Number of samples in training data
        n_iterations_total: Total GSM iterations that were run
        random_seed: Initial random seed used
        output_dir: Directory to save the bundle into
        logger: Logger instance
        max_models: Maximum number of models to include (default: 10)
        ensemble_strategy: 'mean_probability' or 'majority_vote'
        run_name: Optional human-readable experiment label

    Returns:
        Path to the saved .gsm.zip file
    """
    # Select top-K models by F1 score
    sorted_models = sorted(models, key=lambda m: m.f1_score, reverse=True)
    selected_models = sorted_models[:max_models]

    logger.info(
        f"📦 Saving model bundle: {len(selected_models)} models "
        f"(top F1 range: {selected_models[-1].f1_score:.4f}–"
        f"{selected_models[0].f1_score:.4f})"
    )

    # Compute ensemble statistics
    f1_scores = [m.f1_score for m in selected_models]
    auc_scores = [m.auc_roc for m in selected_models]

    # Create bundle ID
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    import re as _re
    _safe_name = _re.sub(r"[^\w\-]", "_", run_name.strip())[:60] if run_name else ""
    _name_suffix = f"_{_safe_name}" if _safe_name else ""
    bundle_id = f"bundle_{dataset_name}_{timestamp}{_name_suffix}"

    # Build metadata
    metadata = BundleMetadata(
        bundle_id=bundle_id,
        created_at=datetime.now().isoformat(),
        gsm_version=GSM_VERSION,
        sklearn_version=sklearn.__version__,
        dataset_name=dataset_name,
        n_training_samples=n_training_samples,
        n_iterations_total=n_iterations_total,
        n_models_saved=len(selected_models),
        random_seed=random_seed,
        feature_names=feature_names,
        group_names=group_names,
        n_features=len(feature_names),
        normalization_method=normalization_method,
        label_mapping=label_mapping,
        ensemble_f1_mean=float(np.mean(f1_scores)),
        ensemble_f1_std=float(np.std(f1_scores)),
        ensemble_auc_mean=float(np.mean(auc_scores)),
        ensemble_auc_std=float(np.std(auc_scores)),
        individual_metrics=[
            {
                "iteration": m.iteration,
                "f1_score": m.f1_score,
                "auc_roc": m.auc_roc,
                "n_features": m.num_features_used,
                "n_groups": m.num_groups_used,
            }
            for m in selected_models
        ],
        ensemble_strategy=ensemble_strategy,
        model_name=model_name,
        run_name=run_name or "",
    )

    # Write to a temp directory, then zip
    output_dir = Path(output_dir)
    bundles_dir = output_dir / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Save metadata
        meta_path = tmp_path / "bundle_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)

        # Save models
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        for i, model_artifact in enumerate(selected_models):
            model_path = models_dir / f"model_{i:03d}_iter{model_artifact.iteration}.joblib"
            joblib.dump(model_artifact.model, model_path, compress=3)

        # Save scaler
        preprocess_dir = tmp_path / "preprocessing"
        preprocess_dir.mkdir()
        joblib.dump(scaler, preprocess_dir / "scaler.joblib", compress=3)

        # Save feature names separately for quick access
        with open(preprocess_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

        # Zip it up
        zip_base = bundles_dir / bundle_id
        archive_path = shutil.make_archive(
            str(zip_base), "zip", root_dir=str(tmp_path)
        )

        # Rename from .zip to .gsm.zip
        final_path = bundles_dir / f"{bundle_id}{BUNDLE_EXTENSION}"
        Path(archive_path).rename(final_path)

    logger.info(f"✅ Model bundle saved: {final_path.name}")
    logger.info(f"   Ensemble F1: {metadata.ensemble_f1_mean:.4f} ± {metadata.ensemble_f1_std:.4f}")
    return final_path


##### Load Bundle #####

def load_bundle(
    bundle_path: Path,
    logger: Optional[logging.Logger] = None,
) -> ModelBundle:
    """Load a ModelBundle from a .gsm.zip file.

    Args:
        bundle_path: Path to the .gsm.zip file
        logger: Optional logger instance

    Returns:
        Fully hydrated ModelBundle ready for inference

    Raises:
        FileNotFoundError: If bundle_path does not exist
        ValueError: If the bundle is corrupted or incompatible
    """
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    if logger:
        logger.info(f"📦 Loading model bundle: {bundle_path.name}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Extract the zip
        shutil.unpack_archive(str(bundle_path), str(tmp_path), "zip")

        # Load metadata
        meta_path = tmp_path / "bundle_metadata.json"
        if not meta_path.exists():
            raise ValueError("Invalid bundle: missing bundle_metadata.json")

        with open(meta_path) as f:
            meta_dict = json.load(f)

        metadata = BundleMetadata(**meta_dict)

        # Warn on sklearn version mismatch
        current_sklearn = sklearn.__version__
        if metadata.sklearn_version != current_sklearn:
            msg = (
                f"⚠️ Bundle was created with sklearn {metadata.sklearn_version}, "
                f"but current version is {current_sklearn}. "
                "Predictions may differ."
            )
            if logger:
                logger.warning(msg)

        # Load scaler
        scaler_path = tmp_path / "preprocessing" / "scaler.joblib"
        if not scaler_path.exists():
            raise ValueError("Invalid bundle: missing preprocessing/scaler.joblib")
        scaler = joblib.load(scaler_path)

        # Load models
        models_dir = tmp_path / "models"
        if not models_dir.exists():
            raise ValueError("Invalid bundle: missing models/ directory")

        model_files = sorted(models_dir.glob("model_*.joblib"))
        if not model_files:
            raise ValueError("Invalid bundle: no model files found")

        # Reconstruct ModelArtifact objects
        model_artifacts = []
        for i, mf in enumerate(model_files):
            model = joblib.load(mf)
            # Match with individual metrics from metadata
            metrics = (
                metadata.individual_metrics[i]
                if i < len(metadata.individual_metrics)
                else {}
            )
            model_artifacts.append(ModelArtifact(
                model=model,
                iteration=metrics.get("iteration", i + 1),
                f1_score=metrics.get("f1_score", 0.0),
                auc_roc=metrics.get("auc_roc", 0.0),
                num_features_used=metrics.get("n_features", metadata.n_features),
                num_groups_used=metrics.get("n_groups", len(metadata.group_names)),
            ))

    if logger:
        logger.info(
            f"✅ Bundle loaded: {metadata.n_models_saved} models, "
            f"{metadata.n_features} features, "
            f"F1={metadata.ensemble_f1_mean:.4f} ± {metadata.ensemble_f1_std:.4f}"
        )

    return ModelBundle(
        metadata=metadata,
        models=model_artifacts,
        scaler=scaler,
    )


##### Bundle Info #####

def bundle_info(bundle_path: Path) -> str:
    """Return a human-readable summary of a model bundle.

    Reads only the metadata JSON from the zip — does NOT
    deserialize the heavy model/scaler files, so this is fast
    even for large bundles.

    Args:
        bundle_path: Path to the .gsm.zip file

    Returns:
        Formatted multi-line string with bundle details
    """
    import zipfile

    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    with zipfile.ZipFile(str(bundle_path), "r") as zf:
        with zf.open("bundle_metadata.json") as f:
            meta_dict = json.load(f)

    meta = BundleMetadata(**meta_dict)

    lines = [
        "═" * 60,
        "  GSM Model Bundle Summary",
        "═" * 60,
        f"  Bundle ID:        {meta.bundle_id}",
        f"  Created:          {meta.created_at}",
    ]
    if meta.run_name:
        lines.append(f"  Experiment Name:  {meta.run_name}")
    lines += [
        f"  GSM Version:      {meta.gsm_version}",
        f"  Sklearn Version:  {meta.sklearn_version}",
        "",
        "  ── Training Info ──",
        f"  Dataset:          {meta.dataset_name}",
        f"  Classifier:       {meta.model_name}",
        f"  Samples:          {meta.n_training_samples}",
        f"  Iterations:       {meta.n_iterations_total}",
        f"  Random Seed:      {meta.random_seed}",
        "",
        "  ── Model Info ──",
        f"  Models Saved:     {meta.n_models_saved}",
        f"  Features:         {meta.n_features}",
        f"  Disease Groups:   {len(meta.group_names)}",
        f"  Ensemble:         {meta.ensemble_strategy}",
        f"  Normalization:    {meta.normalization_method}",
        "",
        "  ── Performance ──",
        f"  Ensemble F1:      {meta.ensemble_f1_mean:.4f} ± {meta.ensemble_f1_std:.4f}",
        f"  Ensemble AUC:     {meta.ensemble_auc_mean:.4f} ± {meta.ensemble_auc_std:.4f}",
        "",
        "  ── Individual Model Metrics ──",
    ]
    for i, m in enumerate(meta.individual_metrics):
        lines.append(
            f"    Model {i+1}: F1={m['f1_score']:.4f}  AUC={m['auc_roc']:.4f}  "
            f"(iter {m['iteration']}, {m.get('n_features', '?')} features)"
        )

    lines.append("")
    lines.append("  ── Top Disease Groups ──")
    for i, gname in enumerate(meta.group_names[:10]):
        lines.append(f"    {i+1}. {gname}")
    if len(meta.group_names) > 10:
        lines.append(f"    ... and {len(meta.group_names) - 10} more")

    lines.append("")
    lines.append("  ── Top Features (first 20) ──")
    for i, fname in enumerate(meta.feature_names[:20]):
        lines.append(f"    {i+1}. {fname}")
    if len(meta.feature_names) > 20:
        lines.append(f"    ... and {len(meta.feature_names) - 20} more")

    # Bundle file size
    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    lines.append("")
    lines.append(f"  File Size:        {size_mb:.2f} MB")
    lines.append("═" * 60)
    return "\n".join(lines)
