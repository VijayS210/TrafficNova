import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

df = pd.read_csv("data/multicity_road_parking_data.csv")

features = ["city_id", "zone_id", "hour", "day_of_week", "is_weekend", "weather_code"]
X = df[features]
y_road = df["congestion_level"]
y_parking = df["parking_occupancy_pct"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_road_train, y_road_test = y_road.iloc[:split_idx], y_road.iloc[split_idx:]
y_park_train, y_park_test = y_parking.iloc[:split_idx], y_parking.iloc[split_idx:]

print("Training Multi-City Road Congestion Model...")
road_model = XGBClassifier(
    n_estimators=160, max_depth=6, learning_rate=0.08, random_state=42, eval_metric="mlogloss"
)
road_model.fit(X_train, y_road_train)
print(f"Road Model Accuracy: {accuracy_score(y_road_test, road_model.predict(X_test)) * 100:.2f}%")

print("Training Multi-City Parking Occupancy Model...")
parking_model = XGBRegressor(
    n_estimators=160, max_depth=6, learning_rate=0.08, random_state=42
)
parking_model.fit(X_train, y_park_train)
park_preds = parking_model.predict(X_test)
print(f"Parking Model MAE: ±{mean_absolute_error(y_park_test, park_preds):.2f}% | R2: {r2_score(y_park_test, park_preds):.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(road_model, "models/road_congestion_model.pkl")
joblib.dump(parking_model, "models/parking_model.pkl")
print("Saved multi-city .pkl models successfully!")