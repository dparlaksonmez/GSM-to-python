"""
🧬 Model Training and Evaluation Module 🧬

This module handles the training and evaluation of machine learning models
using gene features from top-ranked groups.

Key Functions:
-------------
- run_modeling: Main entry point for model training with features from top groups
- select_features_from_top_groups: Selects features from specified number of top groups

Example Usage:
------------
>>> result = run_modeling(train_x, train_y, test_x, test_y, 
                         group_ranks, group_mapping, "RandomForest", logger)
>>> print(f"Model achieved F1 score: {result.f1_score:.4f}")
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

from src.scoring.metrics import MetricsData
from src.grouping.grouping_utils import GroupFeatureMappingData


@dataclass
class ModelingResult:
    """Results from model training and evaluation."""
    model_name: str
    num_groups_used: int
    num_features_used: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    used_features: List[str] = field(default_factory=list)
    used_groups: List[str] = field(default_factory=list)


@dataclass
class SelectedFeaturesResult:
    """Result of feature selection from top groups."""
    selected_features: List[str]
    used_group_names: List[str]


def select_features_from_top_groups(
    group_ranks: List[MetricsData],
    group_feature_mapping: List[GroupFeatureMappingData],
    top_n_groups: int,
    logger: logging.Logger
) -> SelectedFeaturesResult:
    """
    Select features from the top N ranked groups.
    
    Args:
        group_ranks: Groups ranked by performance (highest first)
        group_feature_mapping: Group to feature mapping data
        top_n_groups: Number of top groups to include
        logger: Logger for tracking progress
    
    Returns:
        SelectedFeaturesResult containing selected features and used group names
    """
    # Create a mapping from group name to features for easier lookup
    name_to_mapping = {mapping.group_name: mapping for mapping in group_feature_mapping}
    
    # Calculate how many groups we can actually use
    used_groups = min(top_n_groups, len(group_ranks))
    # logger.info(f"🔍 Selecting features from top {used_groups} groups")
    
    selected_features = []
    used_group_names = []
    
    # Process each top group
    for i in range(used_groups):
        group = group_ranks[i]
        group_name = group.name
        used_group_names.append(group_name)
        
        if group_name in name_to_mapping:
            group_features = name_to_mapping[group_name].feature_list
            # logger.info(f"  - Group {i+1}: '{group_name}' with {len(group_features)} features")
            selected_features.extend(group_features)
        else:
            logger.warning(f"⚠️ Group '{group_name}' not found in group-feature mappings")
    
    # Remove duplicate features
    unique_features = list(set(selected_features))
    # logger.info(f"✅ Selected {len(unique_features)} unique features from {len(used_group_names)} groups")
    
    return SelectedFeaturesResult(
        selected_features=unique_features,
        used_group_names=used_group_names
    )


@dataclass
class ModelTrainingResult:
    """Results from model training and evaluation."""
    model: Any
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]


def train_and_evaluate_model(
    model_name: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    logger: logging.Logger
) -> ModelTrainingResult:
    """
    Train a model and evaluate its performance.
    
    Args:
        model_name: Classifier type to use
        train_x: Training features
        train_y: Training labels
        test_x: Testing features
        test_y: Testing labels
        logger: Logger for tracking progress
        
    Returns:
        ModelTrainingResult with trained model, metrics and feature importance
    """
    # Select model type
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42)
    else:
        logger.error(f"❌ Unsupported model type: {model_name}")
        raise ValueError(f"Unsupported model type: {model_name}")
    
    logger.info(f"🚀 Training {model_name} model"
                f" with {len(train_x)} samples and {len(train_x.columns)} features")
    # Train model
    start_time = time.time()
    model.fit(train_x, train_y)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(test_x)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(test_y, y_pred),
        "precision": precision_score(test_y, y_pred, average='binary', zero_division=0),
        "recall": recall_score(test_y, y_pred, average='binary', zero_division=0),
        "f1": f1_score(test_y, y_pred, average='binary', zero_division=0),
        "training_time": training_time
    }
    
    try:
        # Get feature importance if available
        feature_importance = {}
        if isinstance(model, RandomForestClassifier) and hasattr(model, "feature_importances_"):
            for feature, importance in zip(train_x.columns, model.feature_importances_):
                feature_importance[feature] = float(importance)
        elif isinstance(model, SVC):
            # SVC doesn't have native feature importance
            logger.info("SVM models don't provide direct feature importance scores")
    except Exception as e:
        logger.warning(f"⚠️ Unable to extract feature importance: {str(e)}")
        feature_importance = {}
    
    logger.info(f"✅ Model '{model_name}' trained with accuracy: {metrics['accuracy']:.4f}")
    return ModelTrainingResult(
        model=model,
        metrics=metrics,
        feature_importance=feature_importance
    )


def run_modeling(
    data_train_x: pd.DataFrame,
    data_train_y: pd.Series,
    data_test_x: pd.DataFrame,
    data_test_y: pd.Series,
    group_ranks: List[MetricsData],
    group_feature_mapping: List[GroupFeatureMappingData],
    model_name: str,
    top_n_groups: int,
    logger: logging.Logger
) -> ModelingResult:
    """
    Train and evaluate a model using features from top-ranked groups.
    
    Args:
        data_train_x: Training feature data
        data_train_y: Training labels
        data_test_x: Testing feature data
        data_test_y: Testing labels
        group_ranks: List of groups ranked by performance
        group_feature_mapping: Group to feature mapping data
        model_name: Name of ML model to use
        top_n_groups: Number of top groups to use
        logger: Logger for tracking progress
        
    Returns:
        ModelingResult with model performance metrics
    """
    try:
        # Select features from top groups
        features_result = select_features_from_top_groups(
            group_ranks, 
            group_feature_mapping, 
            top_n_groups,
            logger
        )
        
        # Filter to only include features that exist in the data
        available_features = [f for f in features_result.selected_features if f in data_train_x.columns]
        
        logger.info(f"✅ Found {len(available_features)} valid features for modeling")
        
        if len(available_features) < len(features_result.selected_features):
            missing = len(features_result.selected_features) - len(available_features)
            logger.warning(f"⚠️ {missing} features not found in dataset")
        
        if not available_features:
            logger.error("❌ No valid features available for modeling")
            # Return empty result with zeros
            return ModelingResult(
                model_name=model_name,
                num_groups_used=0,
                num_features_used=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0
            )
        
        # Filter data to use only selected features
        train_x = data_train_x[available_features]
        test_x = data_test_x[available_features]
        
        logger.info(f"🔬 Training {model_name} model with {len(available_features)} features from {len(features_result.used_group_names)} groups")
        
        # Train and evaluate model
        training_result = train_and_evaluate_model(
            model_name,
            train_x,
            data_train_y,
            test_x,
            data_test_y,
            logger
        )
        
        # Create and return result
        return ModelingResult(
            model_name=model_name,
            num_groups_used=len(features_result.used_group_names),
            num_features_used=len(available_features),
            accuracy=training_result.metrics["accuracy"],
            precision=training_result.metrics["precision"],
            recall=training_result.metrics["recall"],
            f1_score=training_result.metrics["f1"],
            training_time=training_result.metrics["training_time"],
            feature_importance=training_result.feature_importance,
            used_features=available_features,
            used_groups=features_result.used_group_names
        )
        
    except Exception as e:
        logger.error(f"❌ Error in modeling: {str(e)}")
        raise

