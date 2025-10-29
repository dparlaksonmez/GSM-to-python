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

from calendar import c
from dataclasses import dataclass
from typing import Dict
import logging
import numpy as np
import pandas as pd
from src.scoring.metrics import MetricsData
from sklearn.model_selection import cross_validate
from src.machine_learning.classification import get_classifier_object
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score


@dataclass
class ScoringParameters:
    """Configuration parameters for scoring gene groups."""
    group_name: str
    data_x: pd.DataFrame
    labels: pd.Series
    classifier_name: str
    logger: logging.Logger
    cross_validation_folds: int = 5
    

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
    if params.data_x.empty or params.labels.empty:
        raise ValueError("❌ Input data or labels are empty")
    
    if params.data_x.shape[0] != params.labels.shape[0]:
        raise ValueError("❌ Number of samples in data and labels don't match")

    try:
        # logger.info(f"🎬 Starting scoring process with {params.classifier_name} classifier")
        # logger.info(f"📊 Input data shape: {params.data_x.shape}")

        # Get classifier object
        classifier = get_classifier_object(params.classifier_name)

        # Define scoring metrics with zero division handling
        scoring_metrics = {
            'accuracy': 'accuracy',
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=1),
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=1),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=1),
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=1),
            'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=1),
            'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=1),
            'balanced_accuracy': 'balanced_accuracy',
            'roc_auc': 'roc_auc_ovo',
            'neg_log_loss': 'neg_log_loss'
        }

        # Perform cross-validation
        scores = cross_validate(
            classifier,
            params.data_x,
            params.labels,
            cv= params.cross_validation_folds,
            scoring=scoring_metrics,
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