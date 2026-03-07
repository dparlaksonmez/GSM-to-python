"""
🏥 GSM Clinical Inference Package

Purpose:
    Provides clinical inference capabilities for the GSM pipeline.
    Enables saving trained model bundles and using them to diagnose
    new patient samples.

Key Modules:
    - model_bundle: ModelBundle dataclass, save/load/inspect
    - inference_engine: Core inference logic with ensemble support
    - clinical_report: Human-readable clinical report generation
"""

from src.inference.model_bundle import ModelBundle, save_bundle, load_bundle
from src.inference.inference_engine import (
    infer, InferenceResult, InferenceSummary,
    multi_infer, MultiBundleResult, MultiBundleSummary,
)

__all__ = [
    "ModelBundle",
    "save_bundle",
    "load_bundle",
    "infer",
    "InferenceResult",
    "InferenceSummary",
    "multi_infer",
    "MultiBundleResult",
    "MultiBundleSummary",
]
