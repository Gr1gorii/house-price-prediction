"""
main.py — runs the full House Price Prediction pipeline end-to-end.

Steps:
  1. Generate synthetic dataset (if not already present)
  2. Clean and process the raw data
  3. Run EDA and save plots to images/
  4. Engineer new features
  5. Split data and train three models
  6. Evaluate and compare models
  7. Save the best model
  8. Show an example prediction

Usage:
  python main.py
"""

import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves plots without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    engineer_features,
    generate_synthetic_data,
    get_feature_columns,
    load_and_clean_data,
)
from src.evaluate_model import (
    evaluate_model,
    plot_actual_vs_predicted,
    plot_feature_importance,
    print_comparison_table,
)
from src.predict import predict_price
from src.train_model import save_model, train_models

# ── Paths ───────────────────────────────────────────────────────────────────────
RAW_DATA   = 'data/raw/house_prices.csv'
PROCESSED  = 'data/processed/house_prices_clean.csv'
MODEL_PATH = 'models/best_model.pkl'
IMAGES_DIR = 'images'


def create_directories():
    for path in ['data/raw', 'data/processed', 'models', IMAGES_DIR]:
        os.makedirs(path, exist_ok=True)


def run_eda(df):
    """Generate and save key EDA plots."""
    print("\n  Generating EDA plots...")

    # 1. Price distribution (normal + log scale)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(df['price'], bins=40, color='steelblue', edgecolor='white')
    axes[0].set_title('Price Distribution')
    axes[0].set_xlabel('Price ($)')
    axes[0].set_ylabel('Count')

    axes[1].hist(np.log1p(df['price']), bins=40, color='coral', edgecolor='white')
    axes[1].set_title('Price Distribution (log scale)')
    axes[1].set_xlabel('log(Price + 1)')
    axes[1].set_ylabel('Count')

    plt.suptitle('House Price Distributions', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/price_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Price vs Area scatter (coloured by rooms)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df['area_sqm'], df['price'],
                    c=df['rooms'], cmap='viridis', alpha=0.45, s=18)
    plt.colorbar(sc, ax=ax, label='Rooms')
    ax.set_xlabel('Area (sqm)')
    ax.set_ylabel('Price ($)')
    ax.set_title('Price vs Area (coloured by number of rooms)')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/price_vs_area.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Average price by district
    avg_price = df.groupby('district')['price'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_price.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white', width=0.65)
    ax.set_title('Average Price by District')
    ax.set_xlabel('District')
    ax.set_ylabel('Average Price ($)')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/price_by_district.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Correlation matrix
    num_cols = [
        'price', 'area_sqm', 'rooms', 'bedrooms', 'bathrooms',
        'floor', 'building_age', 'distance_to_center_km',
        'has_balcony', 'has_parking',
    ]
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                mask=mask, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 9})
    ax.set_title('Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Price by number of rooms (boxplot)
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='price', by='rooms', ax=ax,
               boxprops=dict(color='steelblue'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title('Price Distribution by Number of Rooms')
    plt.suptitle('')
    ax.set_xlabel('Number of Rooms')
    ax.set_ylabel('Price ($)')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/price_by_rooms.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved 5 EDA plots to {IMAGES_DIR}/")


def main():
    print("=" * 58)
    print("   House Price Prediction — ML Project")
    print("=" * 58)

    # 1. Directories
    create_directories()

    # 2. Data generation
    if not os.path.exists(RAW_DATA):
        print("\n[1/7] Generating synthetic dataset...")
        generate_synthetic_data(n_samples=1500, save_path=RAW_DATA)
    else:
        print(f"\n[1/7] Raw data already present: {RAW_DATA}")

    # 3. Data cleaning
    print("\n[2/7] Cleaning data...")
    df = load_and_clean_data(RAW_DATA, PROCESSED)

    # 4. EDA
    print("\n[3/7] Running EDA...")
    run_eda(df)

    # 5. Feature engineering
    print("\n[4/7] Engineering features...")
    df = engineer_features(df)
    print(f"  Dataset shape after feature engineering: {df.shape}")

    # 6. Train / test split
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # 7. Training
    print("\n[5/7] Training models...")
    trained_models = train_models(X_train, y_train)

    # 8. Evaluation
    print("\n[6/7] Evaluating models...")
    results = [
        evaluate_model(model, X_test, y_test, model_name=name)
        for name, model in trained_models.items()
    ]
    comparison = print_comparison_table(results)

    # 9. Save best model (lowest RMSE)
    best_name  = comparison.iloc[0]['Model']
    best_model = trained_models[best_name]
    print(f"\n  Best model: {best_name}")
    save_model(best_model, MODEL_PATH)

    # 10. Save evaluation plots
    print("\n[7/7] Saving evaluation plots...")
    plot_actual_vs_predicted(
        best_model, X_test, y_test,
        model_name=best_name,
        save_path=f'{IMAGES_DIR}/actual_vs_predicted.png',
    )
    plot_feature_importance(
        best_model, feature_cols,
        model_name=best_name,
        save_path=f'{IMAGES_DIR}/feature_importance.png',
    )

    # 11. Example prediction
    print("\n--- Example Prediction ---")
    predict_price(
        area_sqm=70,
        rooms=3,
        bedrooms=2,
        bathrooms=1,
        floor=4,
        total_floors=9,
        building_age=15,
        distance_to_center_km=5.2,
        district="Central",
        has_balcony=1,
        has_parking=0,
        condition="good",
    )

    print("All done! Check images/ for plots and models/ for the saved model.\n")


if __name__ == "__main__":
    main()
