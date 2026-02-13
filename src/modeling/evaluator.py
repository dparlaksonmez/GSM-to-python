# model_evaluator.py

"""
Model Evaluation Module

This module provides functions for evaluating machine learning models in the GSM pipeline.
It includes functionality for:
- Model performance evaluation with comprehensive metrics
- Prediction generation
- Feature importance tracking
- Confidence score calculation

Key Functions:
- evaluate_model: Generate predictions and evaluate model performance
- calculate_metrics: Compute comprehensive classification metrics
- update_feature_ranks: Track and update feature importance rankings
- calculate_confidence_scores: Generate prediction confidence metrics
"""

from src.scoring.metrics import MetricsData

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix
)


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    logger,
    y_pred_proba: Optional[np.ndarray] = None,
) -> MetricsData:
    """
    Calculate comprehensive classification performance metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        logger: Logger instance
        y_pred_proba: Prediction probabilities (optional)
        
    Returns:
        MetricsData object containing all calculated metrics
        
    Raises:
        ValueError: If input shapes don't match
    """
    try:
        if y_true.shape != y_pred.shape:
            raise ValueError("Shape mismatch between y_true and y_pred")
            
        # Initialize metrics object
        metrics = MetricsData(name="model_evaluation")
        
        # Basic metrics
        metrics.accuracy = float(balanced_accuracy_score(y_true, y_pred))
        metrics.precision = float(precision_score(y_true, y_pred))
        metrics.recall = float(recall_score(y_true, y_pred))
        metrics.f1 = float(f1_score(y_true, y_pred))
        
        # Calculate specificity from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.specificity = float(tn / (tn + fp))
        
        # ROC AUC if probabilities are provided
        if y_pred_proba is not None:
            metrics.roc_auc = float(roc_auc_score(y_true, y_pred_proba))
            
            # Calculate confidence metrics
            confidence_scores = calculate_confidence_scores(
                y_pred_proba=y_pred_proba,
                logger=logger
            )
            # Add confidence metrics as additional attributes
            for key, value in confidence_scores.items():
                setattr(metrics, key, value)
            
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Error calculating metrics: {str(e)}")
        raise

def update_feature_ranks(
    current_ranks: pd.DataFrame,
    feature_scores: pd.Series,
    logger,
    method: str = 'mean',
    weights: Optional[List[float]] = None
): #TODO: check return type
    """
    Update feature importance rankings with new scores.
    
    Args:
        current_ranks: Current feature rankings DataFrame
        feature_scores: New feature importance scores
        method: Method to combine scores ('mean', 'max', or 'weighted')
        weights: Weights for weighted averaging (required if method='weighted')
        
    Returns:
        Updated feature rankings DataFrame
        
    Raises:
        ValueError: If invalid method specified or weights missing for weighted method
    """
    if method not in ['mean', 'max', 'weighted']:
        raise ValueError("Method must be one of: 'mean', 'max', 'weighted'")
        
    if method == 'weighted' and weights is None:
        raise ValueError("Weights must be provided for weighted averaging")
        
    try:
        combined = pd.concat([current_ranks, feature_scores])
        
        if method == 'mean':
            return combined.groupby('feature').mean()
        elif method == 'max':
            return combined.groupby('feature').max()
        else:  # weighted
            if len(weights) < len(combined.groupby('feature').first()):
                raise ValueError("Not enough weights for the number of feature groups")
            return combined.groupby('feature').apply(
                lambda x: float(np.average(x, weights=weights[:len(x)]))
            )
    except Exception as e:
        logger.error(f"Error updating feature ranks: {str(e)}")
        raise

def calculate_confidence_scores(
    y_pred_proba: np.ndarray,
    logger,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate confidence metrics for predictions.
    
    Args:
        y_pred_proba: Prediction probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary containing:
        - mean_confidence: Average prediction confidence
        - high_confidence_ratio: Proportion of high-confidence predictions
        - decision_boundary_ratio: Proportion of predictions near decision boundary
    """
    try:
        confidence_scores = np.abs(y_pred_proba - threshold)
        high_conf_mask = confidence_scores >= 0.25
        boundary_mask = confidence_scores <= 0.1
        
        return {
            'mean_confidence': float(np.mean(confidence_scores)),
            'high_confidence_ratio': float(np.mean(high_conf_mask)),
            'decision_boundary_ratio': float(np.mean(boundary_mask))
        }
        
    except Exception as e:
        logger.error(f"Error calculating confidence scores: {str(e)}")
        raise