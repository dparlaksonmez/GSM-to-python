"""
LogitBoost Classifier Module 🧬

Purpose:
    Implementation and interface for the LogitBoost model for gene analysis.
    Uses GradientBoostingClassifier as a LogitBoost-like implementation.

Key Functions:
    - train_logitboost: Function to train the LogitBoost-style model
    - predict_logitboost: Function to make predictions using the trained model

Example Usage:
    model = train_logitboost(X_train, y_train, n_estimators=100)
    predictions = predict_logitboost(model, X_test)

Note:
    This implementation uses sklearn's GradientBoostingClassifier with logistic loss,
    which provides similar functionality to LogitBoost algorithms.
"""

from sklearn.ensemble import GradientBoostingClassifier
from typing import Optional, Literal, Union


def train_logitboost(
    X_train, 
    y_train, 
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    max_features: Optional[Union[int, float, Literal['auto', 'sqrt', 'log2']]] = None,
    random_state: Optional[int] = None
):
    """Train the LogitBoost-style model for gene classification.

    Args:
        X_train: Training features (genes × samples)
        y_train: Training labels
        n_estimators: Number of boosting stages (default: 100)
        learning_rate: Learning rate shrinks contribution of each tree (default: 0.1)
        max_depth: Maximum depth of individual regression estimators (default: 3)
        min_samples_split: Minimum samples required to split internal node (default: 2)
        min_samples_leaf: Minimum samples required to be at leaf node (default: 1)
        subsample: Fraction of samples to use for fitting trees (default: 1.0)
        max_features: Number of features to consider when looking for best split
        random_state: Random state for reproducibility

    Returns:
        Trained LogitBoost-style model

    Example:
        >>> model = train_logitboost(X_train, y_train, n_estimators=150, learning_rate=0.05)
        >>> print(f"LogitBoost model trained with {model.n_estimators} estimators")
    """
    model = GradientBoostingClassifier(
        loss='log_loss',  # Use logistic loss for LogitBoost-like behavior
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model


def predict_logitboost(model, X_test):
    """Make predictions using the trained LogitBoost-style model.

    Args:
        model: Trained LogitBoost-style model
        X_test: Test features for prediction

    Returns:
        Array of predictions

    Example:
        >>> predictions = predict_logitboost(model, X_test)
        >>> print(f"Made predictions for {len(predictions)} samples")
    """
    return model.predict(X_test)


def predict_proba_logitboost(model, X_test):
    """Get prediction probabilities using the trained LogitBoost-style model.

    Args:
        model: Trained LogitBoost-style model
        X_test: Test features for prediction

    Returns:
        Array of prediction probabilities

    Example:
        >>> probabilities = predict_proba_logitboost(model, X_test)
        >>> print(f"Got probabilities for {len(probabilities)} samples")
    """
    return model.predict_proba(X_test)


def get_feature_importance_logitboost(model):
    """Get feature importance from the trained LogitBoost-style model.

    Args:
        model: Trained LogitBoost-style model

    Returns:
        Array of feature importances

    Example:
        >>> importance = get_feature_importance_logitboost(model)
        >>> print(f"Feature importance shape: {importance.shape}")
    """
    return model.feature_importances_


def get_staged_predict_proba_logitboost(model, X_test):
    """Get staged prediction probabilities for model interpretation.

    Args:
        model: Trained LogitBoost-style model
        X_test: Test features for prediction

    Returns:
        Generator yielding prediction probabilities for each stage

    Example:
        >>> staged_probas = list(get_staged_predict_proba_logitboost(model, X_test))
        >>> print(f"Got staged probabilities for {len(staged_probas)} stages")
    """
    return model.staged_predict_proba(X_test)