"""
File Purpose:
This module contains functions for training and predicting with machine learning models.

Key Functions:
- train_model: Trains a specified machine learning model on the provided data.
- predict: Generates predictions using the trained model.

Usage Example:
    from machine_learning.classification import train_model, predict

    model, X_train, y_train, X_test, y_test = train_model('RandomForest', data_x, data_y)
    predictions = predict(model, X_test)
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def train_model(model_name: str, data_x: pd.DataFrame, data_y: pd.Series):
    """
    Train a specified machine learning model on the provided data.

    Args:
        model: The name of the machine learning model to use ('RandomForest' or 'GradientBoosting').
        data_x: Feature DataFrame.
        data_y: Target Series.

    Returns:
        Tuple containing:
        - Trained model
    """

    # Initialize the model
    if model_name == 'RandomForest':
        classifier = RandomForestClassifier()
    elif model_name == 'GradientBoosting':
        classifier = GradientBoostingClassifier()
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    # Train the model
    classifier.fit(data_x, data_y)

    return classifier

def predict(model, X_test: pd.DataFrame):
    """
    Generate predictions using the trained model.

    Args:
        model: The trained machine learning model.
        X_test: Testing features DataFrame.

    Returns:
        Predictions for the test set.
    """
    return model.predict(X_test)

def get_classifier_object(model_name: str, n_estimators: int = 50):
    """
    Get the classifier object based on the model name.

    Args:
        model_name: The name of the machine learning model to use ('RandomForest' or 'GradientBoosting').
        n_estimators: Number of trees/estimators (default: 50, reduced for faster scoring)

    Returns:
        The classifier object.
    
    Note:
        No random_state is set here - the classifier will use the global 
        np.random.seed() which is set per iteration for proper variation.
    """
    if model_name == 'RandomForest':
        # Use fewer trees (50 vs 100 default) - sufficient for group ranking
        # n_jobs=1 to avoid nested parallelism (groups are already parallelized)
        return RandomForestClassifier(n_estimators=n_estimators, n_jobs=1)
    elif model_name == 'GradientBoosting':
        return GradientBoostingClassifier(n_estimators=n_estimators)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
