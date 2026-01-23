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

def score_features(
    data_x: pd.DataFrame,
    labels: pd.Series,
    feature_names: List[str],
    logger: logging.Logger
) -> List[FeatureScore]:
    """
    Score individual features using multiple metrics.

    Args:
        data_x: Feature matrix (samples × features)
        labels: Target labels
        feature_names: List of feature names
        logger: Logger instance

    Returns:
        List of FeatureScore objects for each feature

    Raises:
        ValueError: If input data dimensions don't match
    """
    try:
        if data_x.shape[1] != len(feature_names):
            raise ValueError("Number of features doesn't match feature names")

        logger.info(f"📊 Scoring {len(feature_names)} features...")

        logger.info("⏳ Computing mutual information scores...")
        mutual_info = mutual_info_classif(data_x, labels)

        logger.info("⏳ Training random forest for feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(data_x, labels)
        importance_scores = rf.feature_importances_

        # Score each feature individually
        feature_scores = []
        for idx, feature in tqdm(
            enumerate(feature_names),
            total=len(feature_names),
            desc="📊 Scoring features",
            unit="feature"
        ):
            # Calculate F1 score for binary classification based on this feature
            feature_data = data_x.iloc[:, idx]
            threshold = feature_data.median()
            predictions = (feature_data > threshold).astype(int)
            f1 = f1_score(labels, predictions, zero_division=0)

            feature_scores.append(FeatureScore(
                feature_name=feature,
                f1_score=float(f1),
                importance_score=float(importance_scores[idx]),
                mutual_info=float(mutual_info[idx])
            ))

        logger.info("✅ Feature scoring completed")
        return feature_scores

    except Exception as e:
        logger.error(f"❌ Error during feature scoring: {str(e)}")
        raise

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

        logger.info("✅ Feature importance calculation completed")
        return importance_dict

    except Exception as e:
        logger.error(f"❌ Error calculating feature importance: {str(e)}")
        raise