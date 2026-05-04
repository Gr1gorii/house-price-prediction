import os

import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples=1500, save_path=None):
    """
    Generate a realistic synthetic real estate dataset.
    Price is computed from features with ~10% random noise added.
    """
    np.random.seed(42)

    districts = ['Central', 'Downtown', 'North', 'South', 'East', 'West', 'Suburbs']
    conditions = ['excellent', 'good', 'fair', 'poor']

    # How much each district and condition affects price
    district_mult = {
        'Central': 1.40, 'Downtown': 1.50, 'North': 1.10,
        'South': 0.90,   'East': 1.00,     'West': 1.10, 'Suburbs': 0.75
    }
    condition_mult = {
        'excellent': 1.20, 'good': 1.00, 'fair': 0.85, 'poor': 0.70
    }

    # Basic property features
    area_sqm       = np.round(np.random.uniform(30, 200, n_samples), 1)
    total_floors   = np.random.randint(3, 26, n_samples)
    floor          = np.array([np.random.randint(1, tf + 1) for tf in total_floors])
    rooms          = np.clip(np.floor(area_sqm / 20).astype(int), 1, 6)
    bedrooms       = np.maximum(rooms - 1, 1)
    bathrooms      = np.clip(rooms // 2, 1, 3)
    building_age   = np.random.randint(0, 61, n_samples)
    distance       = np.round(np.random.uniform(0.5, 25.0, n_samples), 1)

    district_arr  = np.random.choice(districts, n_samples,
                                     p=[0.15, 0.10, 0.15, 0.15, 0.15, 0.15, 0.15])
    condition_arr = np.random.choice(conditions, n_samples,
                                     p=[0.20, 0.45, 0.25, 0.10])

    has_balcony = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
    has_parking = np.random.choice([0, 1], n_samples, p=[0.45, 0.55])

    # Compute price from features (multiplicative formula + noise)
    base_price_per_sqm = 2000
    prices = []
    for i in range(n_samples):
        price = (
            area_sqm[i] * base_price_per_sqm
            * district_mult[district_arr[i]]
            * condition_mult[condition_arr[i]]
            * (1 + rooms[i] * 0.03)
            * (1 - building_age[i] * 0.004)
            * max(0.60, 1 - distance[i] * 0.02)
            * (1 + has_balcony[i] * 0.05)
            * (1 + has_parking[i] * 0.07)
        )
        noise = np.random.normal(1.0, 0.10)
        prices.append(max(round(price * noise, -2), 15000))

    df = pd.DataFrame({
        'area_sqm':              area_sqm,
        'rooms':                 rooms,
        'bedrooms':              bedrooms,
        'bathrooms':             bathrooms.astype(float),
        'floor':                 floor.astype(float),
        'total_floors':          total_floors.astype(float),
        'building_age':          building_age.astype(float),
        'distance_to_center_km': distance,
        'district':              district_arr,
        'has_balcony':           has_balcony,
        'has_parking':           has_parking,
        'condition':             condition_arr,
        'price':                 prices,
    })

    # Introduce ~3% missing values in a few columns (realistic)
    for col in ['building_age', 'floor', 'bathrooms']:
        idx = np.random.choice(df.index, size=int(n_samples * 0.03), replace=False)
        df.loc[idx, col] = np.nan

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset saved: {save_path}  ({n_samples} rows)")

    return df


def load_and_clean_data(raw_path, processed_path=None):
    """
    Load raw CSV, handle duplicates, fill missing values, and remove obvious outliers.
    Returns a cleaned DataFrame.
    """
    df = pd.read_csv(raw_path)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # --- Duplicates ---
    n_before = len(df)
    df = df.drop_duplicates()
    removed = n_before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows")

    # --- Missing values ---
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values:")
        print(missing[missing > 0])

    # Fill numeric columns with median / sensible default
    df['building_age'] = df['building_age'].fillna(df['building_age'].median())
    df['floor']        = df['floor'].fillna(df['floor'].median())
    df['bathrooms']    = df['bathrooms'].fillna(1)

    # --- Outlier removal using 1st and 99th percentile ---
    for col, label in [('price', 'price'), ('area_sqm', 'area')]:
        q_low  = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        n_before = len(df)
        df = df[(df[col] >= q_low) & (df[col] <= q_high)]
        print(f"Removed {n_before - len(df)} {label} outliers "
              f"(outside [{q_low:,.0f}, {q_high:,.0f}])")

    df = df.reset_index(drop=True)

    if processed_path:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"\nCleaned data saved: {processed_path}  ({len(df)} rows)")

    return df


def engineer_features(df):
    """
    Add new columns that may help the model.
    Returns a copy of the DataFrame with extra features.
    """
    df = df.copy()

    # How much space each room gets on average
    df['area_per_room'] = (df['area_sqm'] / df['rooms']).round(2)

    # Relative floor position in the building
    df['floor_ratio'] = (df['floor'] / df['total_floors']).round(3)

    # Split building age into categories
    df['building_age_category'] = pd.cut(
        df['building_age'],
        bins=[-1, 5, 20, 40, 200],
        labels=['new', 'modern', 'middle_aged', 'old']
    )

    # Ordinal encoding — districts ordered roughly by avg price level
    district_enc = {
        'Downtown': 7, 'Central': 6, 'North': 5, 'West': 4,
        'East': 3,     'South': 2,   'Suburbs': 1
    }
    df['district_encoded'] = df['district'].map(district_enc)

    # Ordinal encoding — property condition
    condition_enc = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
    df['condition_encoded'] = df['condition'].map(condition_enc)

    # Ordinal encoding — age category
    age_enc = {'new': 4, 'modern': 3, 'middle_aged': 2, 'old': 1}
    df['age_category_encoded'] = df['building_age_category'].map(age_enc).astype(int)

    # Useful for EDA but NOT in training — it's derived from the target (leakage risk)
    df['price_per_sqm'] = (df['price'] / df['area_sqm']).round(2)

    return df


def get_feature_columns():
    """Return the column names used as model input features."""
    return [
        'area_sqm', 'rooms', 'bedrooms', 'bathrooms',
        'floor', 'total_floors', 'building_age', 'distance_to_center_km',
        'has_balcony', 'has_parking',
        'area_per_room', 'floor_ratio',
        'district_encoded', 'condition_encoded', 'age_category_encoded',
    ]
