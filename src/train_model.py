import os

import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


def train_models(X_train, y_train):
    """
    Train three regression models and return them in a dictionary.
    Using random_state=42 everywhere for reproducibility.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def save_model(model, path):
    """Save a trained model to a .pkl file with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved -> {path}")


def load_model(path):
    """Load and return a model from a .pkl file."""
    return joblib.load(path)
