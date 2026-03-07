"""
🎯 Gene Group Scoring Module

This module calculates performance metrics for gene groups using cross-validation.

Key Components:
- ScoringParameters: Configuration for scoring process
- score_data: Main function to evaluate gene groups using ML models

Example:
    params = ScoringParameters(
        data_x=gene_data,
        labels=patient_labels,
        classifier_name='random_forest'
    )
    metrics = score_data(params)
"""

from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
from src.scoring.metrics import MetricsData
from sklearn.model_selection import cross_validate
from src.machine_learning.classification import get_classifier_object
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score


##### CACHED SCORING METRICS (created once, reused) #####
# Pre-create scorer objects to avoid recreation overhead on each call
_SCORING_METRICS = {
    'accuracy': 'accuracy',
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
}


@dataclass
class ScoringParameters:
    """Configuration parameters for scoring gene groups."""
    group_name: str
    data_x: pd.DataFrame
    labels: pd.Series
    classifier_name: str
    logger: logging.Logger
    cross_validation_folds: int = 5
    random_state: int = 42
    

def score_data(params: ScoringParameters) -> MetricsData:
    """
    Calculate performance metrics for a gene group using cross-validation.

    Args:
        params (ScoringParameters): Scoring configuration parameters

    Returns:
        MetricsData: Performance metrics including accuracy, F1, precision, recall

    Raises:
        ValueError: If input data is invalid or empty
        RuntimeError: If scoring process fails
    """
    # Input validation
    # Use len() instead of .empty to support both pandas Series and numpy arrays
    if len(params.data_x) == 0 or len(params.labels) == 0:
        raise ValueError("❌ Input data or labels are empty")
    
    if params.data_x.shape[0] != params.labels.shape[0]:
        raise ValueError("❌ Number of samples in data and labels don't match")

    try:
        # logger.info(f"🎬 Starting scoring process with {params.classifier_name} classifier")
        # logger.info(f"📊 Input data shape: {params.data_x.shape}")

        # Get classifier object (with explicit random_state for reproducibility
        # in parallel worker processes where global np.random.seed is not inherited)
        classifier = get_classifier_object(params.classifier_name, random_state=params.random_state)

        # Convert to numpy arrays for faster sklearn operations
        X = params.data_x.values if hasattr(params.data_x, 'values') else params.data_x
        y = params.labels.values if hasattr(params.labels, 'values') else params.labels

        # Perform cross-validation using cached metrics
        # Only compute metrics we actually use (4 instead of 10)
        scores = cross_validate(
            classifier,
            X,
            y,
            cv=params.cross_validation_folds,
            scoring=_SCORING_METRICS,
            return_train_score=False,
        )

        # Calculate metrics
        metrics = MetricsData(
            accuracy=float(np.mean(scores['test_accuracy'])),
            accuracy_std=float(np.std(scores['test_accuracy'])),
            f1=float(np.mean(scores['test_f1_macro'])),
            f1_std=float(np.std(scores['test_f1_macro'])),
            precision=float(np.mean(scores['test_precision_macro'])),
            precision_std=float(np.std(scores['test_precision_macro'])),
            recall=float(np.mean(scores['test_recall_macro'])),
            recall_std=float(np.std(scores['test_recall_macro']))
        )

        # logger.info("✅ Scoring completed successfully")
        # logger.info(f"📈 Accuracy: {metrics.accuracy:.3f} ± {metrics.accuracy_std:.3f}")

        return metrics

    except Exception as e:
        if params.logger:
            params.logger.error(f"❌ Error during scoring process: {str(e)}")
        raise RuntimeError(f"Scoring process failed: {str(e)}")