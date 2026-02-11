"""
📊 Metrics Module
================
Handles calculation and management of evaluation metrics for the GSM pipeline.

Key Functions:
- calculate_metrics: Core metrics calculation (accuracy, precision, etc)
- calculate_average_scores: Aggregates scores across features/groups
- rank_by_score: Ranks items by specified metric

Example:
    metrics = calculate_metrics(y_true, y_pred)
    print(f"F1 Score: {metrics.f1:.3f}")
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score)


@dataclass
class MetricsData:
    name: Optional[str] = None

    # Basic metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    specificity: Optional[float] = None
    roc_auc: Optional[float] = None

    # Standard deviations
    accuracy_std: Optional[float] = None
    precision_std: Optional[float] = None
    recall_std: Optional[float] = None
    f1_std: Optional[float] = None
    specificity_std: Optional[float] = None
    roc_auc_std: Optional[float] = None

    # Confidence metrics
    mean_confidence: Optional[float] = None
    high_confidence_ratio: Optional[float] = None
    decision_boundary_ratio: Optional[float] = None

from typing import Literal

@dataclass
class MetricParameters:
    """Parameters for metric calculation configuration"""
    average_method: Literal['binary', 'micro', 'macro', 'samples', 'weighted'] = 'binary'
    confidence_threshold: float = 0.8
    handle_warnings: str = 'raise'  # ignore, raise, warn

def validate_metrics_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate input arrays for metric calculation"""
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
    if len(y_true.shape) != 1:
        raise ValueError("Inputs must be 1-dimensional arrays")

def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     logger: Optional[logging.Logger] = None,
                     y_prob: Optional[np.ndarray] = None,
                     params: Optional[MetricParameters] = None
                     ) -> MetricsData:
    """
    Calculate comprehensive ML evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        params: Metric calculation parameters
        
    Returns:
        MetricsData object with calculated metrics
    
    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics.accuracy:.2f}")
    """
    try:
        validate_metrics_inputs(y_true, y_pred)
        params = params or MetricParameters()
        
        metrics = MetricsData(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average=params.average_method)),
            recall=float(recall_score(y_true, y_pred, average=params.average_method)),
            f1=float(f1_score(y_true, y_pred, average=params.average_method))
        )
        
        # Calculate confidence metrics if probabilities provided
        if y_prob is not None:
            metrics.roc_auc = float(roc_auc_score(y_true, y_prob))
            metrics.mean_confidence = float(np.mean(np.max(y_prob, axis=1)))
            metrics.high_confidence_ratio = float(
                np.mean(np.max(y_prob, axis=1) > params.confidence_threshold)
            )
        
        if logger:
            logger.debug(f"📊 Metrics calculated successfully: {metrics}")
        return metrics
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error calculating metrics: {str(e)}")
        raise

def calculate_average_scores(scores: List[MetricsData], logger: logging.Logger) -> MetricsData:
    """
    Calculate average scores with standard deviations across multiple MetricsData objects.
    
    Args:
        scores: List of MetricsData objects to average
        logger: Logger instance for tracking calculation progress
        
    Returns:
        MetricsData object with averaged metrics and standard deviations
        
    Example:
        >>> scores = [metrics1, metrics2, metrics3]  
        >>> avg_metrics = calculate_average_scores(scores, logger)
        >>> print(f"Average F1: {avg_metrics.f1:.2f} ± {avg_metrics.f1_std:.2f}")
    """
    try:
        if not scores:
            raise ValueError("Empty scores list provided")
            
        logger.info("📊 Calculating average metrics across all scores...")
        
        # Initialize result MetricsData
        result = MetricsData()
        
        # Calculate means and standard deviations for each numeric metric
        # Skip non-numeric fields like 'name' (which is a string)
        numeric_fields = [f for f in MetricsData.__dataclass_fields__
                         if f != 'name' and not f.startswith('_')]
        for field in numeric_fields:
            values = [getattr(score, field) for score in scores 
                     if getattr(score, field) is not None]
            
            if values:
                # Calculate mean
                mean_value = float(np.mean(values))
                setattr(result, field, mean_value)
                
                # Calculate standard deviation if applicable
                std_field = f"{field}_std"
                if std_field in MetricsData.__dataclass_fields__:
                    std_value = float(np.std(values))
                    setattr(result, std_field, std_value)
        
        logger.debug(f"✅ Average metrics calculated successfully: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error calculating average metrics: {str(e)}")
        raise


@dataclass
class RankingResult:
    """Stores ranking results for metrics"""
    indices: List[int]          # Original indices
    scores: List[float]         # Score values used for ranking
    metrics: List[MetricsData]  # Original metrics objects in ranked order
    
def rank_by_score(
    metrics_list: List[MetricsData],
    score_type: str = "f1",
    ascending: bool = False,
    logger: Optional[logging.Logger] = None
) -> RankingResult:
    """
    Ranks a list of metrics based on specified score type.
    
    Args:
        metrics_list: List of MetricsData objects to rank
        score_type: Metric to use for ranking (e.g., "f1", "accuracy")
        ascending: Sort in ascending order if True
        logger: Logger instance
    
    Returns:
        RankingResult containing sorted indices, scores and metrics
    
    Example:
        >>> metrics = [metrics1, metrics2, metrics3]
        >>> ranked = rank_by_score(metrics, score_type="f1", logger=logger)
        >>> print(f"Best F1 score: {ranked.scores[0]:.3f}")
    """
    try:
        if not metrics_list:
            raise ValueError("Empty metrics list provided")
            
        if logger:
            logger.debug(f"Ranking {len(metrics_list)} items by {score_type}")
        
        # Extract scores and create index-score pairs
        score_pairs = [
            (i, float(getattr(m, score_type))) 
            for i, m in enumerate(metrics_list)
            if hasattr(m, score_type)
        ]
        
        if not score_pairs:
            raise ValueError(f"No valid scores found for metric: {score_type}")
            
        # Sort by score
        sorted_pairs = sorted(
            score_pairs, 
            key=lambda x: x[1], 
            reverse=not ascending
        )
        
        # Unzip sorted results
        indices, scores = zip(*sorted_pairs)
        indices = list(indices)
        scores = list(scores)
        
        # Get metrics in ranked order
        ranked_metrics = [metrics_list[i] for i in indices]
        
        result = RankingResult(
            indices=indices,
            scores=scores, 
            metrics=ranked_metrics
        )
        
        if logger:
            logger.debug(f"✅ Ranking complete. Top score: {scores[0]:.3f}")
        return result
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error ranking metrics: {str(e)}")
        raise