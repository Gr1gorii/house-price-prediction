"""
predict.py — load the saved best model and estimate a property price.

Run directly:  python src/predict.py
"""

import os

import joblib
import pandas as pd

# Relative to this file, go up one level to reach the project root
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_HERE, '..', 'models', 'best_model.pkl')


# ── Encoding helpers (must match the values in data_preprocessing.py) ──────────

def _encode_district(district):
    enc = {
        'Downtown': 7, 'Central': 6, 'North': 5, 'West': 4,
        'East': 3,     'South': 2,   'Suburbs': 1,
    }
    val = enc.get(district)
    if val is None:
        print(f"Warning: unknown district '{district}'. Defaulting to 'East' (3).")
        return 3
    return val


def _encode_condition(condition):
    enc = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
    val = enc.get(condition.lower())
    if val is None:
        print(f"Warning: unknown condition '{condition}'. Defaulting to 'fair' (2).")
        return 2
    return val


def _encode_age_category(building_age):
    if building_age <= 5:
        return 4   # new
    elif building_age <= 20:
        return 3   # modern
    elif building_age <= 40:
        return 2   # middle_aged
    return 1       # old


# ── Main prediction function ────────────────────────────────────────────────────

def predict_price(
    area_sqm,
    rooms,
    bedrooms,
    bathrooms,
    floor,
    total_floors,
    building_age,
    distance_to_center_km,
    district,
    has_balcony,
    has_parking,
    condition,
    model_path=None,
):
    """
    Predict the price of a property from its features.
    Prints a summary and returns the predicted price as a float.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    model = joblib.load(model_path)

    # Build the same feature vector used during training
    input_data = pd.DataFrame([{
        'area_sqm':              area_sqm,
        'rooms':                 rooms,
        'bedrooms':              bedrooms,
        'bathrooms':             bathrooms,
        'floor':                 floor,
        'total_floors':          total_floors,
        'building_age':          building_age,
        'distance_to_center_km': distance_to_center_km,
        'has_balcony':           has_balcony,
        'has_parking':           has_parking,
        'area_per_room':         area_sqm / rooms,
        'floor_ratio':           floor / total_floors,
        'district_encoded':      _encode_district(district),
        'condition_encoded':     _encode_condition(condition),
        'age_category_encoded':  _encode_age_category(building_age),
    }])

    predicted = model.predict(input_data)[0]

    print("\n" + "=" * 46)
    print("   House Price Prediction")
    print("=" * 46)
    print(f"  Area:             {area_sqm} sqm")
    print(f"  Rooms:            {rooms}  (bedrooms: {bedrooms})")
    print(f"  Bathrooms:        {bathrooms}")
    print(f"  Floor:            {floor} / {total_floors}")
    print(f"  Building age:     {building_age} years")
    print(f"  District:         {district}")
    print(f"  Condition:        {condition}")
    print(f"  Distance center:  {distance_to_center_km} km")
    print(f"  Balcony: {'Yes' if has_balcony else 'No '}   "
          f"Parking: {'Yes' if has_parking else 'No'}")
    print("-" * 46)
    print(f"  Estimated price:  ${predicted:,.0f}")
    print(f"  Price per sqm:    ${predicted / area_sqm:,.0f}")
    print("=" * 46 + "\n")

    return predicted


# ── Example usage ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
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
