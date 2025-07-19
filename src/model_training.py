from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def train_logistic_regression(X_train, y_train):
    """
    Trains a logistic regression model on the training data.

    Args:
        X_train: Feature matrix for training (scaled)
        y_train: Target vector for training

    Returns:
        model: Trained logistic regression model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test data using common classification metrics.

    Args:
        model: Trained classifier
        X_test: Feature matrix for testing
        y_test: True target values for testing

    Prints:
        Accuracy, precision, recall, F1-score, confusion matrix
    """
    y_pred = model.predict(X_test)

    print("Model Evaluation Metrics")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=[0, 1]):
    """
    Plots the confusion matrix as a heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (default: [0, 1])
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    """
    Trains a decision tree classifier on the training data.

    Args:
        X_train: Feature matrix for training
        y_train: Target vector for training
        max_depth: Optional maximum depth of the tree
        random_state: Seed for reproducibility

    Returns:
        model: Trained decision tree classifier
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Trains a Random Forest classifier on the training data.

    Args:
        X_train: Feature matrix for training
        y_train: Target vector for training
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of each tree
        random_state: Seed for reproducibility

    Returns:
        model: Trained Random Forest classifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_importances(model, feature_names, top_n=None):
    """
    Plots a bar chart of feature importances from a fitted Random Forest model.

    Args:
        model: Trained RandomForestClassifier
        feature_names: List of feature names (column names)
        top_n: Optional limit on number of top features to display
    """
    importances = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    if top_n:
        features_df = features_df.head(top_n)

    sns.barplot(data=features_df, x='Importance', y='Feature', palette='Blues_d')
    plt.title("Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()





