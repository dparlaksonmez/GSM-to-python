"""
Feature Scoring Module 🎯

Purpose:
    Calculate performance metrics and importance scores for individual features
    in gene expression data.

Key Functions:
    - score_features: Evaluate individual features using ML metrics
    - calculate_feature_importance: Extract feature importance from trained models

Example:
    >>> scores = score_features(X_train, y_train, feature_names, logger)
    >>> importance = calculate_feature_importance(model, feature_names)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from tqdm import tqdm
import logging

@dataclass
class FeatureScore:
    """Container for feature scoring results.
    
    Attributes:
        feature_name: Name identifier of the feature
        f1_score: F1 classification score (range 0-1) measuring feature predictive power
        importance_score: Feature importance from tree-based models (range 0-1) 
        mutual_info: Mutual information score measuring feature-target dependency
    """
    feature_name: str  # Unique identifier of the feature (e.g. gene name)
    f1_score: float   # Binary classification performance (0-1)
    importance_score: float  # Tree-based importance (0-1) 
    mutual_info: float  # Feature-target mutual information
    
    def __post_init__(self):
        """Validate score ranges after initialization."""
        if not 0 <= self.f1_score <= 1:
            raise ValueError(f"F1 score must be between 0-1, got {self.f1_score}")
        if not 0 <= self.importance_score <= 1:
            raise ValueError(f"Importance score must be between 0-1, got {self.importance_score}")
        if self.mutual_info < 0:
            raise ValueError(f"Mutual info cannot be negative, got {self.mutual_info}")


##### MUTUAL INFORMATION COMPUTATION #####
def _compute_mutual_info_fast(
    data_x: pd.DataFrame,
    labels: pd.Series,
    n_neighbors: int = 3,
    logger=None,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute mutual information scores efficiently.
    
    Mutual Information (MI) measures how much knowing the value of a feature
    reduces uncertainty about the target label. Higher MI = more informative feature.
    
    - MI = 0: Feature is independent of the target (not useful for prediction)
    - MI > 0: Feature shares information with target (useful for classification)
    
    Uses k-nearest neighbors estimation which is faster with fewer neighbors.
    
    Args:
        data_x: Feature matrix (samples × features)
        labels: Target labels
        n_neighbors: Number of neighbors for MI estimation (default: 3, lower = faster)
        logger: Optional logger for progress messages
        random_state: Random seed for reproducibility
        
    Returns:
        Array of mutual information scores for each feature
    """
    if logger:
        logger.debug(f"Computing MI with k={n_neighbors} neighbors")
    
    # Convert to numpy for speed
    X = data_x.values
    y = labels.values
    
    # Compute all at once with fewer neighbors (faster than batching)
    # n_neighbors=3 is much faster than default n_neighbors=5
    # Explicit random_state for reproducibility
    mutual_info = mutual_info_classif(
        X, y,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    
    return mutual_info


def score_features(
    data_x: pd.DataFrame,
    labels: pd.Series,
    feature_names: List[str],
    logger: logging.Logger,
    random_state: int = 42
) -> List[FeatureScore]:
    """
    Score individual features using multiple metrics.

    Args:
        data_x: Feature matrix (samples × features)
        labels: Target labels
        feature_names: List of feature names
        logger: Logger instance
        random_state: Random seed for reproducibility

    Returns:
        List of FeatureScore objects for each feature

    Raises:
        ValueError: If input data dimensions don't match
    """
    try:
        if data_x.shape[1] != len(feature_names):
            raise ValueError("Number of features doesn't match feature names")

        logger.info(f"Feature scoring: {len(feature_names)} features (MI + RF + F1)")

        mutual_info = _compute_mutual_info_fast(data_x, labels, n_neighbors=3, logger=logger, random_state=random_state)

        # Explicit random_state for reproducibility
        # 50 trees is sufficient for ranking features by importance
        rf = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=random_state)
        rf.fit(data_x, labels)
        importance_scores = rf.feature_importances_

        # Vectorized F1 score calculation for all features at once
        f1_scores = _compute_f1_scores_vectorized(data_x, labels)

        # Build feature score objects
        feature_scores = [
            FeatureScore(
                feature_name=feature_names[idx],
                f1_score=float(f1_scores[idx]),
                importance_score=float(importance_scores[idx]),
                mutual_info=float(mutual_info[idx])
            )
            for idx in range(len(feature_names))
        ]

        logger.info("Feature scoring done")
        return feature_scores

    except Exception as e:
        logger.error(f"Feature scoring failed: {str(e)}")
        raise


def _compute_f1_scores_vectorized(data_x: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    """
    Compute F1 scores for all features using vectorized operations.
    
    This replaces the slow per-feature loop with numpy broadcasting.
    
    Args:
        data_x: Feature matrix (samples × features)
        labels: Target labels (binary)
        
    Returns:
        Array of F1 scores for each feature
    """
    # Convert to numpy for speed
    X = data_x.values
    y = labels.values.astype(int)
    
    # Compute median threshold for each feature (column-wise)
    medians = np.median(X, axis=0)
    
    # Predictions: 1 if value > median, else 0 (vectorized across all features)
    predictions = (X > medians).astype(int)
    
    # Compute F1 for each feature using vectorized confusion matrix components
    # True Positives: prediction=1 and label=1
    # False Positives: prediction=1 and label=0
    # False Negatives: prediction=0 and label=1
    y_expanded = y[:, np.newaxis]  # Shape: (n_samples, 1)
    
    tp = np.sum((predictions == 1) & (y_expanded == 1), axis=0)
    fp = np.sum((predictions == 1) & (y_expanded == 0), axis=0)
    fn = np.sum((predictions == 0) & (y_expanded == 1), axis=0)
    
    # F1 = 2 * precision * recall / (precision + recall)
    # F1 = 2 * TP / (2*TP + FP + FN)
    denominator = 2 * tp + fp + fn
    f1_scores = np.where(denominator > 0, 2 * tp / denominator, 0.0)
    
    return f1_scores


from typing import Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def calculate_feature_importance(
    model: Union[RandomForestClassifier, GradientBoostingClassifier],
    feature_names: List[str],
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Extract feature importance scores from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        logger: Logger instance

    Returns:
        Dictionary mapping feature names to importance scores

    Raises:
        AttributeError: If model doesn't support feature importance
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            raise AttributeError("Model doesn't support feature importance calculation")

        importance_dict = dict(zip(feature_names, model.feature_importances_))
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        logger.debug("Feature importance extracted")
        return importance_dict

    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise