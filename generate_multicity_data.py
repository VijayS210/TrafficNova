import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
np.random.seed(42)

# City IDs (0 to 5) and their baseline congestion multipliers
CITIES = {
    0: {"name": "Bengaluru", "traffic_bias": 10, "parking_bias": 8},
    1: {"name": "Hyderabad", "traffic_bias": 5, "parking_bias": 5},
    2: {"name": "Mumbai", "traffic_bias": 12, "parking_bias": 12},
    3: {"name": "Chennai", "traffic_bias": 6, "parking_bias": 6},
    4: {"name": "Delhi", "traffic_bias": 9, "parking_bias": 9},
    5: {"name": "Kolkata", "traffic_bias": 7, "parking_bias": 6},
}

# 5 Zone Archetypes per city (0: Major Bottleneck Junction, 1: Central Business District, 2: Commercial/Retail Hub, 3: IT/Tech Corridor, 4: Transit/Residential Hub)
ZONE_PROFILES = {
    0: {"name": "Major Bottleneck Junction", "t_bias": 22, "slots": 160},
    1: {"name": "Central Business District", "t_bias": 16, "slots": 320},
    2: {"name": "Commercial & Retail Hub", "t_bias": 14, "slots": 240},
    3: {"name": "IT / Corporate Corridor", "t_bias": 20, "slots": 420},
    4: {"name": "Transit & Residential Hub", "t_bias": 12, "slots": 190},
}

dates = pd.date_range(start="2026-01-01", periods=4320, freq="h")
records = []

for dt in dates:
    hour = dt.hour
    dow = dt.dayofweek
    is_weekend = 1 if dow >= 5 else 0
    weather_code = np.random.choice([0, 1, 2], p=[0.65, 0.25, 0.10])

    morning_peak = 1 if (8 <= hour <= 11 and not is_weekend) else 0
    evening_peak = 1 if (17 <= hour <= 21) else 0

    for city_id, c_info in CITIES.items():
        for zone_id, z_info in ZONE_PROFILES.items():
            weekend_mall_rush = (
                1 if (is_weekend and zone_id in [1, 2, 4] and 13 <= hour <= 22) else 0
            )

            traffic_score = (
                14
                + c_info["traffic_bias"]
                + z_info["t_bias"]
                + (morning_peak * 30)
                + (evening_peak * 36)
                + (weekend_mall_rush * 24)
                + (weather_code * 11)
                + np.random.normal(0, 5)
            )
            traffic_score = np.clip(traffic_score, 5, 100)

            congestion_level = 2 if traffic_score >= 70 else (1 if traffic_score >= 42 else 0)

            parking_occ = (
                18
                + c_info["parking_bias"]
                + (morning_peak * 28 if zone_id in [0, 3] else morning_peak * 16)
                + (evening_peak * 34)
                + (weekend_mall_rush * 30)
                + (weather_code * 6)
                + np.random.normal(0, 4)
            )
            parking_occ = round(float(np.clip(parking_occ, 5, 99)), 1)

            records.append(
                {
                    "timestamp": dt,
                    "city_id": city_id,
                    "city_name": c_info["name"],
                    "zone_id": zone_id,
                    "hour": hour,
                    "day_of_week": dow,
                    "is_weekend": is_weekend,
                    "weather_code": weather_code,
                    "congestion_level": congestion_level,
                    "parking_occupancy_pct": parking_occ,
                }
            )

df = pd.DataFrame(records)
df.to_csv("data/multicity_road_parking_data.csv", index=False)
print(f"Saved {len(df)} rows across 6 cities to data/multicity_road_parking_data.csv")