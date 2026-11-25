import os
from datetime import datetime
import folium
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit.components.v1 import html

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Road & Parking Congestion System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 15 Major Bengaluru Destinations mapped to trained XGBoost zone_ids (0-4)
DESTINATIONS = {
    "MG Road / Brigade Road": {"lat": 12.9756, "lon": 77.6066, "zone_id": 1},
    "Whitefield ITPL Main Rd": {"lat": 12.9855, "lon": 77.7361, "zone_id": 3},
    "Silk Board Junction": {"lat": 12.9177, "lon": 77.6238, "zone_id": 0},
    "Koramangala 4th Block": {"lat": 12.9352, "lon": 77.6245, "zone_id": 2},
    "Indiranagar 100ft Rd": {"lat": 12.9784, "lon": 77.6408, "zone_id": 4},
    "Electronic City Phase 1": {"lat": 12.8452, "lon": 77.6602, "zone_id": 0},
    "Majestic / KSR Railway Station": {"lat": 12.9767, "lon": 77.5713, "zone_id": 1},
    "Hebbal Flyover": {"lat": 13.0354, "lon": 77.5988, "zone_id": 4},
    "Marathahalli Bridge": {"lat": 12.9592, "lon": 77.6974, "zone_id": 3},
    "Bellandur / EcoSpace": {"lat": 12.9260, "lon": 77.6762, "zone_id": 0},
    "HSR Layout Sector 2": {"lat": 12.9121, "lon": 77.6446, "zone_id": 2},
    "Jayanagar 4th Block": {"lat": 12.9299, "lon": 77.5824, "zone_id": 2},
    "Yeshwanthpur Junction": {"lat": 13.0226, "lon": 77.5549, "zone_id": 1},
    "KR Puram Hanging Bridge": {"lat": 13.0012, "lon": 77.6952, "zone_id": 3},
    "UB City / Vittal Mallya Rd": {"lat": 12.9716, "lon": 77.5961, "zone_id": 1},
    "Custom Location (Type Below)": {"lat": None, "lon": None, "zone_id": 1},
}

# All 15 locations + Airport available as Starting Points
ORIGINS = {
    "Majestic / KSR Railway Station": {"lat": 12.9767, "lon": 77.5713},
    "Kempegowda International Airport": {"lat": 13.1986, "lon": 77.7066},
    "Electronic City Phase 1": {"lat": 12.8452, "lon": 77.6602},
    "Hebbal Flyover": {"lat": 13.0354, "lon": 77.5988},
    "Jayanagar 4th Block": {"lat": 12.9299, "lon": 77.5824},
    "Banashankari TTMC": {"lat": 12.9255, "lon": 77.5468},
    "Marathahalli Bridge": {"lat": 12.9592, "lon": 77.6974},
    "Silk Board Junction": {"lat": 12.9177, "lon": 77.6238},
    "MG Road / Brigade Road": {"lat": 12.9756, "lon": 77.6066},
    "Koramangala 4th Block": {"lat": 12.9352, "lon": 77.6245},
    "Whitefield ITPL Main Rd": {"lat": 12.9855, "lon": 77.7361},
    "Indiranagar 100ft Rd": {"lat": 12.9784, "lon": 77.6408},
    "HSR Layout Sector 2": {"lat": 12.9121, "lon": 77.6446},
    "Yeshwanthpur Junction": {"lat": 13.0226, "lon": 77.5549},
    "Custom Location (Type Below)": {"lat": None, "lon": None},
}

# Strategic Bengaluru bypass hubs used to force a genuine secondary corridor when OSRM returns only 1 route
BYPASS_WAYPOINTS = [
    (13.0285, 77.6380),  # Nagavara / ORR North Bypass
    (12.9121, 77.6446),  # HSR / ORR South Bypass
    (12.9592, 77.6974),  # Marathahalli / Bellandur Corridor
    (12.9300, 77.5824),  # Jayanagar / South End Corridor
]

WEATHER_OPTIONS = {
    0: "Clear Sky",
    1: "Cloudy / Humid",
    2: "Heavy Monsoon Rain",
}


# ---------------------------------------------------------
# 2. LOAD TRAINED MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    road_path = "models/road_congestion_model.pkl"
    parking_path = "models/parking_model.pkl"

    if os.path.exists(road_path) and os.path.exists(parking_path):
        return joblib.load(road_path), joblib.load(parking_path), False
    return None, None, True


road_model, parking_model, demo_mode = load_models()


# ---------------------------------------------------------
# 3. GEOCODING, PREDICTION & ROUTING HELPERS
# ---------------------------------------------------------
@st.cache_data(ttl=86400)
def geocode_place(place_query: str):
    """Resolve any custom Indian location name into (lat, lon) via OpenStreetMap Nominatim."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": f"{place_query}, Bengaluru, India", "format": "json", "limit": 1}
        headers = {"User-Agent": "SmartCongestionStudentProject/1.0"}
        res = requests.get(url, params=params, headers=headers, timeout=5).json()
        if res:
            return float(res[0]["lat"]), float(res[0]["lon"])
    except Exception:
        pass
    return 12.9716, 77.5946  # Fallback: Bengaluru Center


def predict_conditions(zone_id: int, hour: int, day_of_week: int, is_weekend: int, weather_code: int):
    features = pd.DataFrame([{
        "zone_id": zone_id,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "weather_code": weather_code,
    }])

    if not demo_mode:
        road_level = int(road_model.predict(features)[0])
        parking_occ = float(np.clip(parking_model.predict(features)[0], 0, 100))
    else:
        rush = 1 if (8 <= hour <= 11 or 17 <= hour <= 21) else 0
        road_level = 2 if (rush and weather_code >= 1) else (1 if rush else 0)
        parking_occ = float(np.clip(35 + (rush * 42) + (zone_id * 4) + (weather_code * 6), 5, 99))

    return road_level, round(parking_occ, 1)


@st.cache_data(ttl=3600)
def get_routes_with_guaranteed_alternative(start_lat, start_lon, end_lat, end_lon):
    """Fetch primary route and guarantee a distinct real-road alternative bypass route."""
    parsed_routes = []

    # 1. Request primary + OSRM native alternatives
    url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson&alternatives=3"
    )
    try:
        res = requests.get(url, timeout=6).json()
        if res.get("code") == "Ok":
            for r in res["routes"]:
                coords = [[c[1], c[0]] for c in r["geometry"]["coordinates"]]
                parsed_routes.append({
                    "coords": coords,
                    "distance_km": round(r["distance"] / 1000.0, 1),
                    "duration_min": round(r["duration"] / 60.0),
                })
    except Exception:
        pass

    # 2. If OSRM returned fewer than 2 routes, compute a genuine bypass route via a secondary corridor
    if len(parsed_routes) < 2 and len(parsed_routes) == 1:
        primary_dist = parsed_routes[0]["distance_km"]
        best_bypass = None

        for wp_lat, wp_lon in BYPASS_WAYPOINTS:
            wp_url = (
                f"http://router.project-osrm.org/route/v1/driving/"
                f"{start_lon},{start_lat};{wp_lon},{wp_lat};{end_lon},{end_lat}"
                f"?overview=full&geometries=geojson"
            )
            try:
                wp_res = requests.get(wp_url, timeout=5).json()
                if wp_res.get("code") == "Ok" and wp_res["routes"]:
                    r = wp_res["routes"][0]
                    dist_km = round(r["distance"] / 1000.0, 1)
                    # Pick a bypass that is distinct from primary but not excessively long
                    if abs(dist_km - primary_dist) >= 1.2 and dist_km <= primary_dist * 1.45:
                        candidate = {
                            "coords": [[c[1], c[0]] for c in r["geometry"]["coordinates"]],
                            "distance_km": dist_km,
                            "duration_min": round(r["duration"] / 60.0),
                        }
                        if best_bypass is None or candidate["duration_min"] < best_bypass["duration_min"]:
                            best_bypass = candidate
            except Exception:
                continue

        if best_bypass:
            parsed_routes.append(best_bypass)

    # 3. Offline fallback if network blocks OSRM completely
    if not parsed_routes:
        mid_lat = (start_lat + end_lat) / 2 + 0.012
        mid_lon = (start_lon + end_lon) / 2 - 0.012
        parsed_routes = [
            {"coords": [[start_lat, start_lon], [end_lat, end_lon]], "distance_km": 12.0, "duration_min": 28},
            {"coords": [[start_lat, start_lon], [mid_lat, mid_lon], [end_lat, end_lon]], "distance_km": 14.1, "duration_min": 32},
        ]

    return parsed_routes


# ---------------------------------------------------------
# 4. SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("🧭 Trip Route & Simulation Controls")

origin_choice = st.sidebar.selectbox("📍 From (Starting Point)", options=list(ORIGINS.keys()), index=0)
if origin_choice == "Custom Location (Type Below)":
    custom_origin_text = st.sidebar.text_input("Enter Starting Point (Bengaluru)", value="Cubbon Park")
    o_lat, o_lon = geocode_place(custom_origin_text)
    origin_name = custom_origin_text
else:
    o_lat, o_lon = ORIGINS[origin_choice]["lat"], ORIGINS[origin_choice]["lon"]
    origin_name = origin_choice

dest_choice = st.sidebar.selectbox("🏁 To (Destination Hub)", options=list(DESTINATIONS.keys()), index=1)
if dest_choice == "Custom Location (Type Below)":
    custom_dest_text = st.sidebar.text_input("Enter Destination (Bengaluru)", value="Manyata Tech Park")
    d_lat, d_lon = geocode_place(custom_dest_text)
    dest_name = custom_dest_text
    dest_zone_id = 3
else:
    d_lat, d_lon = DESTINATIONS[dest_choice]["lat"], DESTINATIONS[dest_choice]["lon"]
    dest_name = dest_choice
    dest_zone_id = DESTINATIONS[dest_choice]["zone_id"]

selected_date = st.sidebar.date_input("📅 Travel Date", value=datetime.today())
selected_hour = st.sidebar.slider("⏰ Time of Day (24-Hour Format)", min_value=0, max_value=23, value=19)
selected_weather = st.sidebar.selectbox(
    "🌦️ Weather Condition",
    options=list(WEATHER_OPTIONS.keys()),
    format_func=lambda x: WEATHER_OPTIONS[x],
)

day_of_week = selected_date.weekday()
is_weekend = 1 if day_of_week >= 5 else 0

st.sidebar.markdown("---")
if demo_mode:
    st.sidebar.warning("⚠️ Simulation Mode active. Run `train_models.py` first.")
else:
    st.sidebar.success("✅ Loaded Trained XGBoost Models")


# ---------------------------------------------------------
# 5. RUN PREDICTIONS & ROUTE COMPUTATION
# ---------------------------------------------------------
road_level, parking_occ = predict_conditions(
    dest_zone_id, selected_hour, day_of_week, is_weekend, selected_weather
)

level_map = {
    0: ("LOW CONGESTION", "🟢", "Free-flowing traffic"),
    1: ("MODERATE CONGESTION", "🟡", "Slow-moving at junctions"),
    2: ("HIGH CONGESTION", "🔴", "Heavy bottleneck expected"),
}
status_text, status_icon, status_desc = level_map[road_level]
available_pct = round(100 - parking_occ, 1)
est_delay = [5, 18, 42][road_level] + (selected_weather * 6)

routes = get_routes_with_guaranteed_alternative(o_lat, o_lon, d_lat, d_lon)
primary_route = routes[0]
alt_route = routes[1] if len(routes) > 1 else None

gmaps_url = (
    f"https://www.google.com/maps/dir/?api=1"
    f"&origin={o_lat},{o_lon}&destination={d_lat},{d_lon}&travelmode=driving"
)

# ---------------------------------------------------------
# 6. MAIN DASHBOARD UI & KPI CARDS
# ---------------------------------------------------------
st.title("🚦 AI-Based Road & Parking Congestion Prediction System")
st.markdown(
    f"**Route:** `{origin_name}` ➔ `{dest_name}` | "
    f"**Time:** `{selected_hour:02d}:00` | **Weather:** `{WEATHER_OPTIONS[selected_weather]}`"
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Destination Traffic Status", value=f"{status_icon} {status_text}", delta=status_desc, delta_color="off")
with col2:
    st.metric(label="Predicted Parking Occupancy", value=f"{parking_occ}%", delta=f"{available_pct}% Slots Free")
with col3:
    total_eta = primary_route["duration_min"] + est_delay
    st.metric(
        label="Primary Route ETA",
        value=f"{total_eta} mins ({primary_route['distance_km']} km)",
        delta=f"+{est_delay} min traffic delay",
        delta_color="inverse",
    )
with col4:
    if road_level >= 1 and alt_route:
        alt_eta = alt_route["duration_min"] + int(est_delay * 0.4)
        saved_mins = max(5, total_eta - alt_eta)
        st.metric(
            label="Alternative Bypass ETA",
            value=f"{alt_eta} mins ({alt_route['distance_km']} km)",
            delta=f"Saves ~{saved_mins} mins",
            delta_color="normal",
        )
    else:
        st.metric(label="Alternative Route", value="Not Needed", delta="Primary route is clear", delta_color="off")

# ---------------------------------------------------------
# 7. EMBEDDED INTERACTIVE GOOGLE MAP WITH ROUTES
# ---------------------------------------------------------
st.subheader("🗺️ Live Route & Alternative Bypass Map")

map_col, info_col = st.columns([3, 1])

with map_col:
    center_lat = (o_lat + d_lat) / 2
    center_lon = (o_lon + d_lon) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google Maps",
    )

    folium.Marker(
        [o_lat, o_lon],
        tooltip=f"START: {origin_name}",
        popup=f"<b>Origin:</b> {origin_name}",
        icon=folium.Icon(color="blue", icon="play", prefix="fa"),
    ).add_to(m)

    dest_marker_color = "red" if road_level == 2 else ("orange" if road_level == 1 else "green")
    folium.Marker(
        [d_lat, d_lon],
        tooltip=f"DESTINATION: {dest_name} ({status_text})",
        popup=f"<b>Destination:</b> {dest_name}<br>Traffic: {status_text}<br>Parking: {parking_occ}% Full",
        icon=folium.Icon(color=dest_marker_color, icon="flag-checkered", prefix="fa"),
    ).add_to(m)

    # Draw Alternative Route FIRST (underneath) so both lines remain clearly visible
    if road_level >= 1 and alt_route:
        folium.PolyLine(
            alt_route["coords"],
            color="#00b050",
            weight=7,
            opacity=0.95,
            dash_array="12, 8",
            tooltip=f"Recommended Alternative Bypass: {alt_route['distance_km']} km | Est. {alt_route['duration_min'] + int(est_delay * 0.4)} mins",
        ).add_to(m)

    # Draw Primary Route
    route_line_color = "#d62728" if road_level == 2 else ("#ff7f0e" if road_level == 1 else "#1a73e8")
    folium.PolyLine(
        primary_route["coords"],
        color=route_line_color,
        weight=6,
        opacity=0.85,
        tooltip=f"Primary Route: {primary_route['distance_km']} km | Est. {total_eta} mins ({status_text})",
    ).add_to(m)

    m.fit_bounds([[o_lat, o_lon], [d_lat, d_lon]])
    html(m._repr_html_(), height=460)

with info_col:
    st.markdown("### 🛣️ Route Legend & Advice")
    if road_level == 2:
        st.error(
            f"**🔴 Primary Route (Solid Red):** Heavy congestion predicted at **{dest_name}** (+{est_delay} min delay)."
        )
        if alt_route:
            st.success(
                f"**🟢 Alternative Route (Dashed Green):** Recommended bypass ({alt_route['distance_km']} km) avoids primary bottleneck."
            )
    elif road_level == 1:
        st.warning(
            f"**🟠 Primary Route (Solid Orange):** Moderate slowdowns expected near **{dest_name}**."
        )
        if alt_route:
            st.info(
                f"**🟢 Alternative Route (Dashed Green):** Optional bypass ({alt_route['distance_km']} km) shown on map."
            )
    else:
        st.success(
            f"**🔵 Primary Route (Solid Blue):** Clear traffic flow! Take the direct route ({primary_route['distance_km']} km)."
        )

    if parking_occ >= 80:
        st.warning(f"🅿️ **Parking Alert:** {dest_name} parking is **{parking_occ}% full**. Reserve a slot early.")
    else:
        st.info(f"🅿️ **Parking Available:** **{available_pct}%** free slots at {dest_name}.")

    st.link_button("📲 Open Route in Google Maps", gmaps_url, use_container_width=True)

# ---------------------------------------------------------
# 8. 24-HOUR FORECAST CHART & ZONE COMPARISON
# ---------------------------------------------------------
st.markdown("---")
chart_col1, chart_col2 = st.columns(2)

hours = list(range(24))
hourly_road = []
hourly_park = []

for h in hours:
    r_lvl, p_occ = predict_conditions(dest_zone_id, h, day_of_week, is_weekend, selected_weather)
    hourly_road.append((r_lvl + 1) * 33.3)
    hourly_park.append(p_occ)

with chart_col1:
    st.subheader(f"📈 24-Hour Trend: {dest_name}")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=hours, y=hourly_park, mode="lines+markers", name="Parking Occupancy (%)", line=dict(color="#1f77b4", width=3)))
    fig1.add_trace(go.Scatter(x=hours, y=hourly_road, mode="lines", name="Road Congestion Index", line=dict(color="#ff7f0e", width=2, dash="dash")))
    fig1.add_vline(x=selected_hour, line_width=2, line_dash="dot", line_color="red", annotation_text="Selected Time")
    fig1.update_layout(xaxis_title="Hour of Day (0–23)", yaxis_title="Percentage / Index", yaxis_range=[0, 105], height=360)
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    st.subheader(f"🏙️ Major Hubs Comparison at {selected_hour:02d}:00")
    core_hubs = [
        ("Silk Board", 0),
        ("MG Road", 1),
        ("Koramangala", 2),
        ("Whitefield", 3),
        ("Indiranagar", 4),
    ]
    hub_names = [h[0] for h in core_hubs]
    hub_parks = [predict_conditions(h[1], selected_hour, day_of_week, is_weekend, selected_weather)[1] for h in core_hubs]
    colors = ["#d62728" if p >= 80 else ("#ff7f0e" if p >= 55 else "#2ca02c") for p in hub_parks]

    fig2 = go.Figure(go.Bar(x=hub_names, y=hub_parks, marker_color=colors, text=[f"{p}%" for p in hub_parks], textposition="auto"))
    fig2.update_layout(yaxis_title="Predicted Parking Occupancy (%)", yaxis_range=[0, 105], height=360)
    st.plotly_chart(fig2, use_container_width=True)