"""
Decision Tree Classifier Module 🧬

Purpose:
    Implementation and interface for the Decision Tree model for gene analysis.

Key Functions:
    - train_decision_tree: Function to train the Decision Tree model
    - predict_decision_tree: Function to make predictions using the trained model

Example Usage:
    model = train_decision_tree(X_train, y_train, max_depth=10)
    predictions = predict_decision_tree(model, X_test)
"""

from sklearn.tree import DecisionTreeClassifier
from typing import Optional, Literal, Union


def train_decision_tree(
    X_train, 
    y_train, 
    *,
    criterion: Literal['gini', 'entropy', 'log_loss'] = 'gini',
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Optional[Union[int, float, Literal['auto', 'sqrt', 'log2']]] = None,
    random_state: Optional[int] = None
):
    """Train the Decision Tree model for gene classification.

    Args:
        X_train: Training features (genes × samples)
        y_train: Training labels
        criterion: Function to measure quality of split ('gini', 'entropy', 'log_loss')
        max_depth: Maximum depth of the tree (default: None - unlimited)
        min_samples_split: Minimum samples required to split internal node (default: 2)
        min_samples_leaf: Minimum samples required to be at leaf node (default: 1)
        max_features: Number of features to consider when looking for best split
        random_state: Random state for reproducibility

    Returns:
        Trained Decision Tree model

    Example:
        >>> model = train_decision_tree(X_train, y_train, max_depth=5, criterion='entropy')
        >>> print(f"Decision tree trained with max depth: {model.max_depth}")
    """
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model


def predict_decision_tree(model, X_test):
    """Make predictions using the trained Decision Tree model.

    Args:
        model: Trained Decision Tree model
        X_test: Test features for prediction

    Returns:
        Array of predictions

    Example:
        >>> predictions = predict_decision_tree(model, X_test)
        >>> print(f"Made predictions for {len(predictions)} samples")
    """
    return model.predict(X_test)


def predict_proba_decision_tree(model, X_test):
    """Get prediction probabilities using the trained Decision Tree model.

    Args:
        model: Trained Decision Tree model
        X_test: Test features for prediction

    Returns:
        Array of prediction probabilities

    Example:
        >>> probabilities = predict_proba_decision_tree(model, X_test)
        >>> print(f"Got probabilities for {len(probabilities)} samples")
    """
    return model.predict_proba(X_test)


def get_feature_importance_decision_tree(model):
    """Get feature importance from the trained Decision Tree model.

    Args:
        model: Trained Decision Tree model

    Returns:
        Array of feature importances

    Example:
        >>> importance = get_feature_importance_decision_tree(model)
        >>> print(f"Feature importance shape: {importance.shape}")
    """
    return model.feature_importances_