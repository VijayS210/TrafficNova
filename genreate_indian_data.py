import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
np.random.seed(42)

# Indian Urban Zones (Bengaluru Hubs)
zones = {
    0: "Silk Board Junction",
    1: "MG Road / Brigade Road",
    2: "Koramangala 4th Block",
    3: "Whitefield ITPL Main Rd",
    4: "Indiranagar 100ft Rd",
}

# Generate 6 months of hourly data (4,320 hours x 5 zones = 21,600 rows)
dates = pd.date_range(start="2026-01-01", periods=4320, freq="h")
records = []

for dt in dates:
    hour = dt.hour
    dow = dt.dayofweek
    is_weekend = 1 if dow >= 5 else 0
    # Weather: 0 = Clear, 1 = Cloudy/Humid, 2 = Heavy Monsoon Rain
    weather_code = np.random.choice([0, 1, 2], p=[0.65, 0.25, 0.10])

    for zone_id, zone_name in zones.items():
        # Indian peak hours: 8:30 AM - 11:30 AM & 5:30 PM - 9:00 PM
        morning_peak = 1 if (8 <= hour <= 11 and not is_weekend) else 0
        evening_peak = 1 if (17 <= hour <= 21) else 0
        weekend_mall_rush = (
            1 if (is_weekend and zone_id in [1, 2, 4] and 13 <= hour <= 22) else 0
        )

        # Zone-specific congestion baseline
        zone_bias = [22, 16, 14, 20, 12][zone_id]

        # Traffic index (0 to 100)
        traffic_score = (
            18
            + zone_bias
            + (morning_peak * 32)
            + (evening_peak * 38)
            + (weekend_mall_rush * 25)
            + (weather_code * 11)
            + np.random.normal(0, 5)
        )
        traffic_score = np.clip(traffic_score, 5, 100)

        # Target 1: Road Congestion Level (0 = Low, 1 = Moderate, 2 = High)
        if traffic_score >= 70:
            congestion_level = 2
        elif traffic_score >= 42:
            congestion_level = 1
        else:
            congestion_level = 0

        # Target 2: Parking Occupancy Rate (%)
        total_slots = [150, 300, 220, 400, 180][zone_id]
        parking_occ = (
            22
            + (morning_peak * 30 if zone_id in [0, 3] else morning_peak * 18)
            + (evening_peak * 35)
            + (weekend_mall_rush * 32)
            + (weather_code * 6)
            + np.random.normal(0, 4)
        )
        parking_occ = round(float(np.clip(parking_occ, 5, 99)), 1)
        occupied_slots = int((parking_occ / 100.0) * total_slots)

        records.append(
            {
                "timestamp": dt,
                "zone_id": zone_id,
                "zone_name": zone_name,
                "hour": hour,
                "day_of_week": dow,
                "is_weekend": is_weekend,
                "weather_code": weather_code,
                "avg_speed_kmph": round(max(8.0, 55.0 - (traffic_score * 0.42)), 1),
                "congestion_level": congestion_level,
                "total_parking_slots": total_slots,
                "occupied_slots": occupied_slots,
                "parking_occupancy_pct": parking_occ,
            }
        )

df = pd.DataFrame(records)
df.to_csv("data/indian_road_parking_data.csv", index=False)
print(f"Saved {len(df)} rows to data/indian_road_parking_data.csv")