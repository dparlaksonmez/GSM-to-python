"""
AdaBoost Classifier Module 🧬

Purpose:
    Implementation and interface for the AdaBoost model for gene analysis.

Key Functions:
    - train_adaboost: Function to train the AdaBoost model
    - predict_adaboost: Function to make predictions using the trained model

Example Usage:
    model = train_adaboost(X_train, y_train, n_estimators=50)
    predictions = predict_adaboost(model, X_test)
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Optional, Literal


def train_adaboost(
    X_train, 
    y_train, 
    *,
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    algorithm: Literal['SAMME', 'SAMME.R'] = 'SAMME',
    random_state: Optional[int] = None
):
    """Train the AdaBoost model for gene classification.

    Args:
        X_train: Training features (genes × samples)
        y_train: Training labels
        n_estimators: Number of weak learners (default: 50)
        learning_rate: Learning rate shrinks contribution of each classifier (default: 1.0)
        algorithm: Boosting algorithm to use ('SAMME' or 'SAMME.R', default: 'SAMME')
        random_state: Random state for reproducibility

    Returns:
        Trained AdaBoost model

    Example:
        >>> model = train_adaboost(X_train, y_train, n_estimators=100, learning_rate=0.8)
        >>> print(f"Model trained with {model.n_estimators} estimators")
    """
    # Use DecisionTreeClassifier with max_depth=1 as base estimator (decision stumps)
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm=algorithm,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model


def predict_adaboost(model, X_test):
    """Make predictions using the trained AdaBoost model.

    Args:
        model: Trained AdaBoost model
        X_test: Test features for prediction

    Returns:
        Array of predictions

    Example:
        >>> predictions = predict_adaboost(model, X_test)
        >>> print(f"Made predictions for {len(predictions)} samples")
    """
    return model.predict(X_test)


def predict_proba_adaboost(model, X_test):
    """Get prediction probabilities using the trained AdaBoost model.

    Args:
        model: Trained AdaBoost model
        X_test: Test features for prediction

    Returns:
        Array of prediction probabilities

    Example:
        >>> probabilities = predict_proba_adaboost(model, X_test)
        >>> print(f"Got probabilities for {len(probabilities)} samples")
    """
    return model.predict_proba(X_test)