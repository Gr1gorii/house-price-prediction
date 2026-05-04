import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Compute MAE, RMSE, and R2 for a model on the test set."""
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return {
        'Model': model_name,
        'MAE':   round(mae, 0),
        'RMSE':  round(rmse, 0),
        'R2':    round(r2, 4),
    }


def print_comparison_table(results):
    """Print a formatted model comparison table sorted by RMSE."""
    df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)

    print("\n" + "=" * 62)
    print("  Model Comparison (sorted by RMSE, lower is better)")
    print("=" * 62)
    print(df.to_string(index=False))
    print("=" * 62)

    return df


def plot_actual_vs_predicted(model, X_test, y_test, model_name, save_path=None):
    """Scatter plot: actual prices vs model predictions."""
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color='steelblue',
               edgecolors='none', s=22)

    # Perfect-prediction diagonal
    lo = min(float(y_test.min()), float(y_pred.min())) * 0.95
    hi = max(float(y_test.max()), float(y_pred.max())) * 1.05
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')

    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.set_title(f'Actual vs Predicted — {model_name}')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved -> {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, model_name, save_path=None):
    """Horizontal bar chart of feature importances (tree-based models only)."""
    if not hasattr(model, 'feature_importances_'):
        print(f"[skip] {model_name} does not have feature importances")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(importances)))
    importances.plot(kind='barh', ax=ax, color=colors)
    ax.set_title(f'Feature Importance — {model_name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved -> {save_path}")
    plt.close()
